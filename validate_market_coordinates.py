# -*- coding: utf-8 -*-
"""
Auditoría automática de coordenadas de mercados SIPSA.
- No modifica ni reemplaza coordenadas.
- Valida campos, rangos aproximados de Colombia, duplicados y plausibilidad frente
  a resultados de OpenStreetMap Nominatim.
- Produce data/coordinate_validation_report.csv.
- Si Nominatim falla o no da una coincidencia interpretable, marca NO VERIFICABLE
  o REVISAR; nunca declara una coordenada correcta solo por estar dentro de Colombia.
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
COORDS_FILE = DATA_DIR / "market_coordinates.csv"
REPORT_FILE = DATA_DIR / "coordinate_validation_report.csv"
USER_AGENT = os.getenv(
    "AGROCLIMA_GEOCODER_USER_AGENT",
    "AgroClimaIntelligenceColombia/1.0 (coordinate quality audit)"
)
# Do not query Nominatim on every row if the workflow is run repeatedly.
CACHE_FILE = ROOT / ".runtime" / "nominatim_validation_cache.csv"

# Broad bounding box for mainland Colombia; this is a screening test, not proof of accuracy.
LAT_MIN, LAT_MAX = -5.5, 14.5
LON_MIN, LON_MAX = -83.5, -65.0
MAX_DISTANCE_KM = float(os.getenv("AGROCLIMA_COORD_MAX_DISTANCE_KM", "5"))
ENABLE_ONLINE = os.getenv("AGROCLIMA_COORD_ONLINE_CHECK", "1").strip().lower() not in {"0", "false", "no"}

def norm(value) -> str:
    return " ".join(str(value or "").strip().lower().split())

def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0088
    p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
    dp = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return radius * 2 * math.asin(min(1.0, math.sqrt(a)))

def main():
    if not COORDS_FILE.exists():
        raise FileNotFoundError(f"No existe {COORDS_FILE}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(COORDS_FILE, dtype={"mercado": str})
    required = {"mercado", "latitud", "longitud"}
    if not required.issubset(df.columns):
        raise ValueError(f"El CSV debe contener las columnas: {', '.join(sorted(required))}")

    df["lat_num"] = pd.to_numeric(df["latitud"], errors="coerce")
    df["lon_num"] = pd.to_numeric(df["longitud"], errors="coerce")
    df["mercado_key"] = df["mercado"].map(norm)
    duplicate_name = df["mercado_key"].duplicated(keep=False)
    duplicate_coord = df.assign(coord_key=list(zip(df["lat_num"].round(5), df["lon_num"].round(5))))["coord_key"].duplicated(keep=False)

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if CACHE_FILE.exists():
        try:
            cdf = pd.read_csv(CACHE_FILE).fillna("")
            for _, row in cdf.iterrows():
                cache[str(row.get("query_key", ""))] = row.to_dict()
        except Exception:
            cache = {}

    geolocator = None
    if ENABLE_ONLINE:
        try:
            from geopy.geocoders import Nominatim
            geolocator = Nominatim(user_agent=USER_AGENT, timeout=15)
        except Exception as exc:
            print(f"Geocodificación no disponible: {exc}")

    report = []
    cache_rows = list(cache.values())
    last_request = 0.0

    for i, row in df.iterrows():
        name = str(row.get("mercado", "") or "").strip()
        lat, lon = row["lat_num"], row["lon_num"]
        issues, status = [], "VÁLIDO"
        if not name:
            issues.append("Nombre de mercado vacío")
        if pd.isna(lat) or pd.isna(lon):
            issues.append("Latitud/longitud vacía o no numérica")
            status = "ERROR"
        else:
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                issues.append("Coordenadas fuera de los límites geográficos globales")
                status = "ERROR"
            elif not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
                issues.append("Coordenadas fuera del rango territorial aproximado de Colombia")
                status = "ERROR"
        if duplicate_name.iloc[i]:
            issues.append("Nombre de mercado duplicado")
            status = "REVISAR"
        if duplicate_coord.iloc[i] and not pd.isna(lat) and not pd.isna(lon):
            issues.append("Coordenada repetida con otra fila; confirmar si es intencional")
            status = "REVISAR"

        precision = str(row.get("precision_geografica", "") or "")
        source = str(row.get("fuente_coordenada", "") or "")
        if any(term in norm(precision) for term in ["aproxim", "verificar", "confirmar", "revisar", "pendiente"]):
            issues.append("La propia ficha describe la coordenada como aproximada o pendiente de confirmar")
            if status == "VÁLIDO":
                status = "REVISAR"

        osm_name, osm_lat, osm_lon, distance, osm_type, online_note = "", None, None, None, "", ""
        query_key = norm(name) + "|colombia"
        if status != "ERROR" and ENABLE_ONLINE and geolocator is not None:
            cached = cache.get(query_key)
            if cached and str(cached.get("status", "")) == "OK":
                osm_name = str(cached.get("display_name", ""))
                osm_type = str(cached.get("osm_type", ""))
                try:
                    osm_lat, osm_lon = float(cached["lat"]), float(cached["lon"])
                    distance = haversine_km(lat, lon, osm_lat, osm_lon)
                    online_note = "Resultado reutilizado de caché Nominatim"
                except Exception:
                    osm_lat = osm_lon = distance = None
            else:
                try:
                    # Nominatim public service policy: keep at least 1 second between requests.
                    wait = 1.05 - (time.time() - last_request)
                    if last_request and wait > 0:
                        time.sleep(wait)
                    result = geolocator.geocode(f"{name}, Colombia", exactly_one=True, addressdetails=True, country_codes="co")
                    last_request = time.time()
                    if result is None:
                        online_note = "Nominatim no devolvió resultado para el nombre completo"
                        issues.append(online_note)
                        if status == "VÁLIDO":
                            status = "NO VERIFICABLE"
                    else:
                        osm_name = str(result.address)
                        osm_lat, osm_lon = float(result.latitude), float(result.longitude)
                        osm_type = str((getattr(result, "raw", {}) or {}).get("type", ""))
                        distance = haversine_km(lat, lon, osm_lat, osm_lon)
                        cache[query_key] = {
                            "query_key": query_key, "status": "OK", "display_name": osm_name,
                            "lat": osm_lat, "lon": osm_lon, "osm_type": osm_type
                        }
                        cache_rows = list(cache.values())
                        online_note = "Comparación con resultado Nominatim; no constituye verificación oficial"
                except Exception as exc:
                    last_request = time.time()
                    online_note = f"Fallo de consulta Nominatim: {type(exc).__name__}"
                    issues.append("No se pudo consultar Nominatim; validación externa incompleta")
                    if status == "VÁLIDO":
                        status = "NO VERIFICABLE"

            if distance is not None and distance > MAX_DISTANCE_KM:
                issues.append(f"El resultado de Nominatim queda a {distance:.1f} km; revisar posible discrepancia")
                status = "REVISAR"
            elif distance is not None and distance <= MAX_DISTANCE_KM:
                issues.append(f"Resultado Nominatim a {distance:.1f} km; coincidencia espacial aproximada, no confirmación definitiva")
                if status == "VÁLIDO":
                    status = "REVISAR"  # Nominatim is not authoritative enough to auto-certify a market site.

        if not issues:
            issues.append("Sin errores automáticos básicos; confirmar que el punto corresponde al mercado SIPSA exacto")
            status = "REVISAR"

        report.append({
            "mercado": name,
            "latitud": lat if not pd.isna(lat) else "",
            "longitud": lon if not pd.isna(lon) else "",
            "estado": status,
            "distancia_nominatim_km": round(distance, 3) if distance is not None else "",
            "resultado_nominatim": osm_name,
            "tipo_osm": osm_type,
            "fuente_coordenada": source,
            "precision_geografica": precision,
            "observaciones": " | ".join(dict.fromkeys(issues)),
            "nota_consulta": online_note,
        })

    pd.DataFrame(report).to_csv(REPORT_FILE, index=False, encoding="utf-8-sig")
    if cache_rows:
        pd.DataFrame(cache_rows).drop_duplicates("query_key", keep="last").to_csv(CACHE_FILE, index=False, encoding="utf-8-sig")
    result_df = pd.DataFrame(report)
    counts = result_df["estado"].value_counts().to_dict()
    print(f"Informe generado: {REPORT_FILE}")
    print("Resumen:", counts)
    print("La auditoría no cambia las coordenadas ni certifica automáticamente la ubicación real del mercado.")

if __name__ == "__main__":
    main()
