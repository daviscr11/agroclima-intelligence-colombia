# -*- coding: utf-8 -*-
"""Auditoría mejorada de coordenadas de mercados SIPSA.

Contrasta dos evidencias independientes de OpenStreetMap/Nominatim:
1) búsqueda por un nombre de instalación/mercado específico;
2) geocodificación inversa de las coordenadas exactas ya guardadas.

No corrige ni reemplaza coordenadas. Genera data/coordinate_validation_report.csv y
conserva en data/coordinate_validation_cache.json respuestas exitosas con caducidad.
La coincidencia automática es una señal para priorizar la revisión, no una certificación.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
COORDS_FILE = DATA_DIR / "market_coordinates.csv"
REPORT_FILE = DATA_DIR / "coordinate_validation_report.csv"
CACHE_FILE = DATA_DIR / "coordinate_validation_cache.json"
BASE_URL = "https://nominatim.openstreetmap.org"
USER_AGENT = os.getenv(
    "AGROCLIMA_GEOCODER_USER_AGENT",
    "AgroClimaIntelligenceColombia/1.1 (+https://github.com/DavisCR/agroclima-intelligence-colombia)",
)
ONLINE_CHECK = os.getenv("AGROCLIMA_COORD_ONLINE_CHECK", "1").strip().lower() not in {"0", "false", "no"}
REQUEST_INTERVAL_SECONDS = max(1.05, float(os.getenv("AGROCLIMA_COORD_REQUEST_INTERVAL", "1.1")))
CACHE_TTL_DAYS = int(os.getenv("AGROCLIMA_COORD_CACHE_TTL_DAYS", "90"))
MAX_FORWARD_DISTANCE_KM = float(os.getenv("AGROCLIMA_COORD_MAX_DISTANCE_KM", "1.0"))

# Rango amplio de cribado territorial para mercados en Colombia continental.
# Estar dentro de este rango NO implica que el punto sea el mercado correcto.
LAT_MIN, LAT_MAX = -5.5, 14.5
LON_MIN, LON_MAX = -83.5, -65.0

# Un alias específico evita que una búsqueda de ciudad (p. ej. "Arauca") termine
# seleccionando una población vecina. Son consultas de búsqueda, nunca coordenadas de reemplazo.
MARKET_RULES: list[dict[str, Any]] = [
    {"key": "arauca arauca", "query": "Plaza de Mercado de Arauca, Arauca, Colombia", "facility_terms": ["plaza de mercado de arauca", "plaza de mercado", "mercado de arauca"], "localities": ["arauca"]},
    {"key": "armenia mercar", "query": "MERCAR Central Mayorista de Armenia, Quindio, Colombia", "facility_terms": ["mercar", "central mayorista de armenia"], "localities": ["armenia"]},
    {"key": "barranquilla granabastos", "query": "GranAbastos, Soledad, Atlantico, Colombia", "facility_terms": ["granabastos", "gran abastos", "gran central de abastos del caribe"], "localities": ["soledad", "barranquilla"]},
    {"key": "barranquilla barranquillita", "query": "Mercado Barranquillita, Barranquilla, Atlantico, Colombia", "facility_terms": ["barranquillita", "mercado barranquillita"], "localities": ["barranquilla"]},
    {"key": "bogota d c corabastos", "query": "Corabastos, Bogota, Colombia", "facility_terms": ["corabastos"], "localities": ["bogota"]},
    {"key": "bucaramanga centroabastos", "query": "Centroabastos, Bucaramanga, Santander, Colombia", "facility_terms": ["centroabastos", "centro abastos", "central de abastos de bucaramanga"], "localities": ["bucaramanga"]},
    {"key": "buenaventura valle del cauca", "query": "Galeria de Pueblo Nuevo Plaza de Mercado Jose Hilario Lopez, Buenaventura, Valle del Cauca, Colombia", "facility_terms": ["galeria de pueblo nuevo", "pueblo nuevo", "jose hilario lopez", "plaza de mercado"], "localities": ["buenaventura"]},
    {"key": "cali cavasa", "query": "CAVASA Central de Abastecimientos del Valle del Cauca, Candelaria, Colombia", "facility_terms": ["cavasa", "central de abastecimientos del valle"], "localities": ["candelaria", "cali"]},
    {"key": "cartagena bazurto", "query": "Mercado de Bazurto, Cartagena, Bolivar, Colombia", "facility_terms": ["bazurto", "mercado de bazurto"], "localities": ["cartagena"]},
    {"key": "cartago valle del cauca", "query": "Plaza de Mercado de Cartago, Valle del Cauca, Colombia", "facility_terms": ["plaza de mercado de cartago", "mercado de cartago", "plaza de mercado"], "localities": ["cartago"]},
    {"key": "chiquinquira boyaca", "query": "Plaza de Mercado de Chiquinquira, Boyaca, Colombia", "facility_terms": ["plaza de mercado de chiquinquira", "mercado de chiquinquira", "plaza de mercado"], "localities": ["chiquinquira"]},
    {"key": "cucuta cenabastos", "query": "Cenabastos, Cucuta, Norte de Santander, Colombia", "facility_terms": ["cenabastos", "central de abastos de cucuta"], "localities": ["cucuta"]},
]

# Tipos que normalmente apuntan a un objeto diferente a una plaza/central mayorista.
# Son señales de revisión, no una decisión definitiva: el objeto OSM más cercano puede ser vecino.
NON_MARKET_TYPES = {
    "bank", "school", "university", "college", "clothes", "waste_transfer_station",
    "hospital", "clinic", "place_of_worship", "fuel", "restaurant", "cafe",
    "fast_food", "hotel", "pharmacy", "police", "post_office", "kindergarten",
}
APPROX_WORDS = ("aproxim", "verificar", "confirmar", "revisar", "pendiente", "asociada a", "centro aproximado")


def norm(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    value = unicodedata.normalize("NFKD", str(value).strip().lower())
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
    dp = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return radius * 2 * math.asin(min(1.0, math.sqrt(a)))


def get_rule(market_name: str) -> dict[str, Any] | None:
    key = norm(market_name)
    for rule in MARKET_RULES:
        if norm(rule["key"]) == key:
            return rule
    # Nombre de SIPSA puede cambiar ligeramente; usa coincidencia de tokens distintivos.
    for marker in ("granabastos", "barranquillita", "corabastos", "centroabastos", "cavasa", "bazurto", "cenabastos", "mercar"):
        if marker in key:
            return next((r for r in MARKET_RULES if marker in norm(r["key"]) or marker in norm(r["query"])), None)
    return None


def address_text(item: dict[str, Any] | None) -> str:
    if not isinstance(item, dict):
        return ""
    pieces = [str(item.get("name", "")), str(item.get("display_name", ""))]
    address = item.get("address") or {}
    if isinstance(address, dict):
        pieces.extend(str(v) for v in address.values() if v)
    namedetails = item.get("namedetails") or {}
    if isinstance(namedetails, dict):
        pieces.extend(str(v) for v in namedetails.values() if v)
    return " ".join(pieces)


def locality_text(item: dict[str, Any] | None) -> str:
    if not isinstance(item, dict):
        return ""
    address = item.get("address") or {}
    if not isinstance(address, dict):
        return ""
    keys = ("city", "town", "municipality", "county", "state_district", "city_district", "village", "suburb", "state")
    return ", ".join(str(address[k]) for k in keys if address.get(k))


def item_coords(item: dict[str, Any] | None) -> tuple[float, float] | None:
    if not isinstance(item, dict):
        return None
    try:
        lat, lon = float(item["lat"]), float(item["lon"])
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon
    except (KeyError, TypeError, ValueError):
        pass
    return None


class NominatimClient:
    """Cliente secuencial, con User-Agent identificable, caché y límite global de solicitudes."""
    def __init__(self, cache_file: Path):
        self.cache_file = cache_file
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "es"})
        self.last_request_at = 0.0
        self.blocked_reason = ""
        self.cache: dict[str, dict[str, Any]] = {"entries": {}}
        if cache_file.exists():
            try:
                loaded = json.loads(cache_file.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and isinstance(loaded.get("entries"), dict):
                    self.cache = loaded
            except (OSError, json.JSONDecodeError):
                print("Aviso: caché JSON ilegible; se iniciará una caché nueva.")
        else:
            # Asegura que el paso git add del workflow no falle si el proveedor está caído.
            self._save_cache()

    def _save_cache(self) -> None:
        self.cache["version"] = 2
        self.cache["source"] = "OpenStreetMap Nominatim; datos sujetos a ODbL"
        self.cache["updated_at"] = datetime.now(timezone.utc).isoformat()
        tmp = self.cache_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.cache_file)

    def fetch(self, key: str, endpoint: str, params: dict[str, Any]) -> tuple[Any, str]:
        now = time.time()
        entry = self.cache["entries"].get(key)
        if isinstance(entry, dict):
            try:
                age_days = (now - float(entry.get("fetched_at", 0))) / 86400
                ttl = CACHE_TTL_DAYS if entry.get("status") == "ok" else min(7, CACHE_TTL_DAYS)
                if age_days <= ttl:
                    return entry.get("data"), "caché" 
            except (TypeError, ValueError):
                pass
        if self.blocked_reason:
            return None, f"Consulta en línea omitida después de fallo previo: {self.blocked_reason}"
        delay = REQUEST_INTERVAL_SECONDS - (time.monotonic() - self.last_request_at)
        if self.last_request_at and delay > 0:
            time.sleep(delay)
        try:
            response = self.session.get(f"{BASE_URL}/{endpoint}", params=params, timeout=20)
            self.last_request_at = time.monotonic()
            if response.status_code in (403, 429):
                self.blocked_reason = f"Nominatim devolvió HTTP {response.status_code}; se detuvieron las consultas para respetar el servicio"
                return None, self.blocked_reason
            response.raise_for_status()
            data = response.json()
            # Search devuelve una lista; reverse devuelve un objeto o {"error": ...}.
            ok = bool(data) and not (isinstance(data, dict) and data.get("error"))
            self.cache["entries"][key] = {
                "status": "ok" if ok else "not_found",
                "fetched_at": time.time(),
                "data": data,
            }
            self._save_cache()
            return (data if ok else None), ("consulta nueva" if ok else "Nominatim no encontró resultado")
        except (requests.RequestException, ValueError) as exc:
            self.last_request_at = time.monotonic()
            self.blocked_reason = f"fallo de conexión/respuesta ({type(exc).__name__})"
            return None, f"Fallo Nominatim: {type(exc).__name__}; se detuvieron las consultas restantes"

    def search(self, query: str) -> tuple[list[dict[str, Any]], str]:
        key = "search|" + norm(query)
        data, note = self.fetch(key, "search", {
            "q": query, "format": "jsonv2", "addressdetails": 1, "namedetails": 1,
            "limit": 5, "countrycodes": "co",
        })
        return (data if isinstance(data, list) else []), note

    def reverse(self, lat: float, lon: float) -> tuple[dict[str, Any] | None, str]:
        # Cinco decimales conserva precisión de pocos metros y permite reutilizar caché.
        key = f"reverse|{float(lat):.5f}|{float(lon):.5f}"
        data, note = self.fetch(key, "reverse", {
            "lat": f"{float(lat):.7f}", "lon": f"{float(lon):.7f}",
            "format": "jsonv2", "addressdetails": 1, "namedetails": 1, "zoom": 18,
        })
        return (data if isinstance(data, dict) else None), note


def choose_search_candidate(candidates: list[dict[str, Any]], rule: dict[str, Any], lat: float, lon: float) -> dict[str, Any] | None:
    """Escoge por identidad del establecimiento y cercanía; no toma simplemente el primer resultado."""
    if not candidates:
        return None
    identity_terms = [norm(x) for x in rule.get("facility_terms", []) if norm(x)]
    locality_terms = [norm(x) for x in rule.get("localities", []) if norm(x)]
    scored = []
    for item in candidates:
        text_norm = norm(address_text(item))
        name_match = any(term in text_norm for term in identity_terms)
        locality_match = any(term in text_norm for term in locality_terms)
        coords = item_coords(item)
        distance = haversine_km(lat, lon, *coords) if coords else None
        score = (8 if name_match else 0) + (2 if locality_match else 0)
        if distance is not None:
            score += 3 if distance <= 0.25 else 2 if distance <= 1 else 1 if distance <= 5 else 0
        scored.append((score, -(distance if distance is not None else 999999), item))
    scored.sort(key=lambda v: (v[0], v[1]), reverse=True)
    return scored[0][2]


def inspect_market(name: str, lat: float, lon: float, source: str, precision: str,
                   client: NominatimClient | None, duplicate_name: bool, duplicate_coord: bool) -> dict[str, Any]:
    issues: list[str] = []
    status = "REVISAR"
    if not name:
        issues.append("Nombre de mercado vacío")
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        issues.append("Coordenadas fuera de los límites geográficos globales")
        return {"estado": "ERROR", "nivel_evidencia": "nulo", "observaciones": " | ".join(issues)}
    if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
        issues.append("Coordenadas fuera del rango territorial aproximado de Colombia")
        return {"estado": "ERROR", "nivel_evidencia": "nulo", "observaciones": " | ".join(issues)}
    if duplicate_name:
        issues.append("Nombre de mercado duplicado en el CSV")
    if duplicate_coord:
        issues.append("La coordenada se repite en otra fila; confirmar si es intencional")
    precision_norm = norm(precision)
    flagged_approx = any(norm(word) in precision_norm for word in APPROX_WORDS)
    if flagged_approx:
        issues.append("La ficha declara que el punto es aproximado o necesita confirmación")

    rule = get_rule(name)
    search_item = None
    reverse_item = None
    search_note = reverse_note = "Consulta en línea desactivada"
    search_distance = reverse_distance = None
    search_text = search_category = search_type = search_locality = ""
    reverse_text = reverse_category = reverse_type = reverse_locality = ""
    identity_match = locality_match = reverse_locality_match = False
    reverse_unrelated_near = False
    reverse_suspicious_type = False
    any_online_evidence = False

    if client is not None:
        if rule is None:
            search_note = "No hay alias configurado para este nombre; se evita una búsqueda ambigua"
            issues.append("Añadir alias de instalación específico para mejorar la búsqueda por nombre")
        else:
            candidates, search_note = client.search(rule["query"])
            search_item = choose_search_candidate(candidates, rule, lat, lon)
            if search_item:
                any_online_evidence = True
                search_text = str(search_item.get("display_name") or search_item.get("name") or "")
                search_category = str(search_item.get("category") or "")
                search_type = str(search_item.get("type") or "")
                search_locality = locality_text(search_item)
                search_coords = item_coords(search_item)
                if search_coords:
                    search_distance = haversine_km(lat, lon, *search_coords)
                candidate_norm = norm(address_text(search_item))
                identity_match = any(norm(term) in candidate_norm for term in rule.get("facility_terms", []))
                locality_match = any(norm(term) in candidate_norm for term in rule.get("localities", []))
                if not identity_match:
                    issues.append("La búsqueda por nombre no encontró una coincidencia clara con la instalación esperada")
                if search_distance is not None and search_distance > MAX_FORWARD_DISTANCE_KM:
                    issues.append(f"El resultado de búsqueda de la instalación está a {search_distance:.2f} km del punto guardado")
                if not locality_match:
                    issues.append("La localidad esperada no aparece claramente en el resultado de búsqueda")
            else:
                issues.append("No se encontró candidato por nombre de instalación en Nominatim")

        reverse_item, reverse_note = client.reverse(lat, lon)
        if reverse_item:
            any_online_evidence = True
            reverse_text = str(reverse_item.get("display_name") or reverse_item.get("name") or "")
            reverse_category = str(reverse_item.get("category") or "")
            reverse_type = str(reverse_item.get("type") or "")
            reverse_locality = locality_text(reverse_item)
            reverse_coords = item_coords(reverse_item)
            if reverse_coords:
                reverse_distance = haversine_km(lat, lon, *reverse_coords)
            if rule:
                reverse_locality_match = any(norm(term) in norm(reverse_text + " " + reverse_locality) for term in rule.get("localities", []))
                if not reverse_locality_match:
                    issues.append("La dirección inversa no confirma claramente la localidad esperada")
            if reverse_type in NON_MARKET_TYPES:
                reverse_suspicious_type = True
                reverse_unrelated_near = reverse_distance is not None and reverse_distance <= 0.15
                if reverse_unrelated_near:
                    issues.append(f"En las coordenadas exactas, Nominatim devuelve un objeto de tipo '{reverse_type}', no una central/plaza mayorista")
                else:
                    issues.append(f"El objeto más cercano figura como '{reverse_type}'; puede ser un predio vecino y requiere revisión")
        else:
            issues.append("No hubo resultado de geocodificación inversa para las coordenadas exactas")

    # Nunca convertir el mero rango territorial en una validación positiva.
    duplicate_problem = duplicate_name or duplicate_coord
    if client is None:
        status, evidence = "NO VERIFICABLE", "solo controles geométricos"
        issues.append("La comprobación en línea no se ejecutó; no puede verificarse el lugar real")
    elif not any_online_evidence:
        status, evidence = "NO VERIFICABLE", "sin evidencia externa"
        issues.append("No hay evidencia externa suficiente para pronunciarse sobre la ubicación")
    elif identity_match and search_distance is not None and search_distance <= 0.25 and locality_match \
            and reverse_item is not None and reverse_locality_match \
            and not reverse_suspicious_type and not reverse_unrelated_near \
            and not flagged_approx and not duplicate_problem:
        status, evidence = "COINCIDENCIA PROBABLE", "alta, pero no certificada"
        issues.append("Nombre de instalación, localidad y proximidad son compatibles; confirmar visualmente sobre mapa")
    else:
        status, evidence = "REVISAR", "parcial o contradictoria"
        if reverse_unrelated_near:
            issues.append("Prioridad alta: revisar el marcador exacto sobre cartografía")
        elif identity_match and search_distance is not None and search_distance <= 1:
            issues.append("La instalación aparece cerca, pero no hay evidencia suficiente para validar el punto exacto")
        elif not identity_match:
            issues.append("La validación automática no pudo vincular inequívocamente el punto con el mercado SIPSA")

    return {
        "estado": status,
        "nivel_evidencia": evidence,
        "distancia_busqueda_km": round(search_distance, 3) if search_distance is not None else "",
        "nombre_busqueda_osm": search_text,
        "categoria_busqueda_osm": search_category,
        "tipo_busqueda_osm": search_type,
        "coincide_nombre_instalacion": "sí" if identity_match else "no" if search_item else "sin resultado",
        "localidad_busqueda_osm": search_locality,
        "distancia_objeto_reverse_km": round(reverse_distance, 3) if reverse_distance is not None else "",
        "lugar_cercano_a_coordenada": reverse_text,
        "categoria_reverse_osm": reverse_category,
        "tipo_reverse_osm": reverse_type,
        "localidad_reverse_osm": reverse_locality,
        "coincide_localidad_reverse": "sí" if reverse_locality_match else "no" if reverse_item and rule else "sin resultado",
        "consulta_busqueda": rule["query"] if rule else "",
        "nota_consulta_busqueda": search_note,
        "nota_consulta_inversa": reverse_note,
        "observaciones": " | ".join(dict.fromkeys(issues)),
    }


def main() -> None:
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
    duplicate_names = df["mercado_key"].duplicated(keep=False)
    coord_keys = list(zip(df["lat_num"].round(5), df["lon_num"].round(5)))
    duplicate_coords = pd.Series(coord_keys).duplicated(keep=False)

    client = NominatimClient(CACHE_FILE) if ONLINE_CHECK else None
    rows = []
    for i, row in df.iterrows():
        market = str(row.get("mercado", "") or "").strip()
        lat, lon = row["lat_num"], row["lon_num"]
        source = str(row.get("fuente_coordenada", "") or "")
        precision = str(row.get("precision_geografica", "") or "")
        if pd.isna(lat) or pd.isna(lon):
            audit = {"estado": "ERROR", "nivel_evidencia": "nulo", "observaciones": "Latitud/longitud vacía o no numérica"}
        else:
            audit = inspect_market(market, float(lat), float(lon), source, precision, client,
                                   bool(duplicate_names.iloc[i]), bool(duplicate_coords.iloc[i]))
        base = {
            "mercado": market,
            "latitud": lat if not pd.isna(lat) else "",
            "longitud": lon if not pd.isna(lon) else "",
            "fuente_coordenada": source,
            "precision_geografica": precision,
        }
        rows.append({**base, **audit, "atribucion_fuente": "OpenStreetMap contributors / Nominatim; consulta de auditoría, no fuente oficial de coordenadas"})
        print(f"[{i + 1}/{len(df)}] {market}: {audit.get('estado', 'REVISAR')}")

    result = pd.DataFrame(rows)
    # Ordenar severidad para que los problemas más relevantes salgan al principio del CSV.
    order = {"ERROR": 0, "REVISAR": 1, "NO VERIFICABLE": 2, "COINCIDENCIA PROBABLE": 3}
    result["_orden"] = result["estado"].map(order).fillna(9)
    result = result.sort_values(["_orden", "mercado"]).drop(columns="_orden")
    result.to_csv(REPORT_FILE, index=False, encoding="utf-8-sig")
    counts = result["estado"].value_counts().to_dict()
    print(f"\nInforme generado: {REPORT_FILE}")
    print("Resumen:", counts)
    print(f"Caché persistente: {CACHE_FILE}")
    print("Las coordenadas originales NO se han modificado. Una coincidencia probable no equivale a verificación oficial.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR en auditoría de coordenadas: {exc}", file=sys.stderr)
        raise
