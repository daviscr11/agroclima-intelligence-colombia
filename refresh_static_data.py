# -*- coding: utf-8 -*-
"""Pipeline para producir el JSON que consume el Static Space.

Se ejecuta en GitHub Actions (no dentro de Hugging Face Static). Solo acepta fuentes
reales. Si alguna descarga, validación geográfica o estimación falla, termina con error
y conserva el último JSON publicado.
"""
from __future__ import annotations

import itertools
import json
import os
import re
import sys
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from dane_auto import DANE_CATALOG_URL, download_sipsa_archives
from sipsa_audit_report import SipsaAuditCollector, assert_selected_products_scale
from agroclima_spvarx_real import (
    norm_text,
    load_sipsa_data,
    prepare_prices,
    fetch_nasa_power,
    build_knn_weights,
    build_spvarx_panel,
    estimate_spvarx,
    compare_spatial_model_families,
    model_stability,
    compute_irfs,
    stationarity_audit,
    residual_moran,
    rolling_spvarx_backtest,
    spatial_weights_sensitivity,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RUNTIME_DIR = ROOT / ".runtime"
RAW_DIR = RUNTIME_DIR / "sipsa_raw"
CLIMATE_CACHE = RUNTIME_DIR / "nasa_power_monthly.csv"
COORDS_FILE = DATA_DIR / "market_coordinates.csv"
OUTPUT_JSON = DATA_DIR / "observatorio.json"
BACKTEST_CSV = DATA_DIR / "spvarx_rolling_backtest.csv"
DEPARTMENTS_GEOJSON = DATA_DIR / "colombia_departments.geojson"
DEPARTMENTS_GEOJSON_URL = "https://raw.githubusercontent.com/ytolosa/mapas-colombia/main/output/geojson/col_departamentos_real_nacional.geojson"

# Coordenadas verificadas para mercados cuyo nombre SIPSA no se resuelve bien
# con geocodificación automática. Mercar corresponde a la Central Mayorista
# de Armenia (Quindío); ubicación aproximada basada en OpenStreetMap/Mapcarta.
# Fuente de referencia: https://mapcarta.com/es/W293128586
KNOWN_MARKET_COORDINATES = {
    "armenia mercar": {
        "latitud": 4.5283,
        "longitud": -75.7133,
        "nombre_geocodificado": "Mercar - Central Mayorista de Armenia, Quindío",
        "fuente_coordenada": "OpenStreetMap vía Mapcarta: https://mapcarta.com/es/W293128586",
        "precision_geografica": "instalación mayorista aproximada; verificar",
    },
    "armenia central mayorista": {
        "latitud": 4.5283,
        "longitud": -75.7133,
        "nombre_geocodificado": "Mercar - Central Mayorista de Armenia, Quindío",
        "fuente_coordenada": "OpenStreetMap vía Mapcarta: https://mapcarta.com/es/W293128586",
        "precision_geografica": "instalación mayorista aproximada; verificar",
    },
    "barranquilla granabastos": {
        "latitud": 10.88766,
        "longitud": -74.79877,
        "nombre_geocodificado": "GranAbastos, Soledad, Atlántico",
        "fuente_coordenada": "OpenStreetMap vía Mapcarta: https://mapcarta.com/W201507633; dirección contrastada con https://www.granabastos.com.co/landing/conocenos",
        "precision_geografica": "complejo mayorista aproximado; verificar",
    },
    "granabastos barranquilla": {
        "latitud": 10.88766,
        "longitud": -74.79877,
        "nombre_geocodificado": "GranAbastos, Soledad, Atlántico",
        "fuente_coordenada": "OpenStreetMap vía Mapcarta: https://mapcarta.com/W201507633; dirección contrastada con https://www.granabastos.com.co/landing/conocenos",
        "precision_geografica": "complejo mayorista aproximado; verificar",
    },
    "soledad granabastos": {
        "latitud": 10.88766,
        "longitud": -74.79877,
        "nombre_geocodificado": "GranAbastos, Soledad, Atlántico",
        "fuente_coordenada": "OpenStreetMap vía Mapcarta: https://mapcarta.com/W201507633; dirección contrastada con https://www.granabastos.com.co/landing/conocenos",
        "precision_geografica": "complejo mayorista aproximado; verificar",
    },
}

START_YEAR = int(os.getenv("AGROCLIMA_START_YEAR", "2019"))
END_YEAR = int(os.getenv("AGROCLIMA_END_YEAR", "2024"))
MIN_MARKETS = int(os.getenv("AGROCLIMA_MIN_MARKETS", "6"))
MIN_MONTHS = int(os.getenv("AGROCLIMA_MIN_MONTHS", "36"))
MAX_CANDIDATE_PRODUCTS = int(os.getenv("AGROCLIMA_MAX_CANDIDATE_PRODUCTS", "24"))
MAX_MARKETS = int(os.getenv("AGROCLIMA_MAX_MARKETS", "12"))
KNN_K = int(os.getenv("AGROCLIMA_KNN_K", "4"))
KNN_SENSITIVITY_VALUES = tuple(dict.fromkeys(
    [int(value.strip()) for value in os.getenv("AGROCLIMA_KNN_SENSITIVITY", "3,4,5").split(",") if value.strip()]
    + [KNN_K]
))
IRF_HORIZON = int(os.getenv("AGROCLIMA_IRF_HORIZON", "18"))
IRF_BOOTSTRAP_REPS = int(os.getenv("AGROCLIMA_IRF_BOOTSTRAP_REPS", "399"))
IRF_BOOTSTRAP_SEED = int(os.getenv("AGROCLIMA_IRF_BOOTSTRAP_SEED", "20261025"))
BACKTEST_MIN_TRAIN_MONTHS = int(os.getenv("AGROCLIMA_BACKTEST_MIN_TRAIN_MONTHS", "36"))


def json_safe(value: Any) -> Any:
    """Convierte tipos numpy/pandas a JSON estricto y representa faltantes como null."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray, pd.Index)):
        return [json_safe(v) for v in list(value)]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def discover_product_variants(raw: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> list[dict[str, Any]]:
    """Detecta variantes reales (producto-unidad-variedad) con suficiente cobertura."""
    d = raw[raw["fecha"].between(start, end)].copy()
    for col in ["producto", "unidad", "variedad", "mercado"]:
        d[col] = d[col].fillna("").astype(str).str.strip()
    d = d[(d["producto"] != "") & (d["unidad"] != "") & (d["precio"] > 0)].copy()
    d["mercado_key"] = d["mercado"].map(norm_text)
    d["producto_key"] = d["producto"].map(norm_text)
    d["unidad_key"] = d["unidad"].map(norm_text)
    d["variedad_key"] = d["variedad"].map(norm_text)

    # No tratar una fila sin variedad como serie distinta si ese producto sí tiene variedades explícitas.
    product_has_varieties = d.groupby("producto_key")["variedad_key"].apply(lambda x: any(v for v in x.unique()))
    candidates: list[dict[str, Any]] = []
    keys = ["producto_key", "unidad_key", "variedad_key"]
    for _, g in d.groupby(keys, dropna=False, sort=False):
        product = g["producto"].mode().iloc[0]
        unit = g["unidad"].mode().iloc[0]
        variety = g["variedad"].mode().iloc[0] if g["variedad"].astype(str).str.strip().ne("").any() else ""
        pkey = norm_text(product)
        if product_has_varieties.get(pkey, False) and not norm_text(variety):
            continue
        monthly = g.groupby(["mercado_key", "fecha"], as_index=False)["precio"].mean()
        if monthly["fecha"].nunique() < MIN_MONTHS:
            continue
        months_market = monthly.groupby("mercado_key")["fecha"].nunique()
        eligible = int((months_market >= MIN_MONTHS).sum())
        if eligible < MIN_MARKETS:
            continue
        candidates.append({
            "producto": str(product), "unidad": str(unit), "variedad": str(variety),
            "mercados_con_cobertura": eligible,
            "meses_observados": int(monthly["fecha"].nunique()),
            "observaciones": int(len(g)),
            "score": int(eligible * monthly["fecha"].nunique()),
        })

    # Conservar una variante con mejor cobertura por producto para evitar duplicar endógenas.
    candidates.sort(key=lambda x: (x["score"], x["observaciones"]), reverse=True)
    one_per_product: dict[str, dict[str, Any]] = {}
    for item in candidates:
        one_per_product.setdefault(norm_text(item["producto"]), item)
    return list(one_per_product.values())[:MAX_CANDIDATE_PRODUCTS]


def choose_best_pair(raw: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> tuple[list[dict[str, str]], pd.DataFrame, pd.DataFrame, list[str], pd.DatetimeIndex, dict[str, Any]]:
    """Elige dos productos reales con mayor cobertura conjunta completa; no interpola faltantes."""
    candidates = discover_product_variants(raw, start_date, end_date)
    if len(candidates) < 2:
        raise RuntimeError(
            f"Solo encontré {len(candidates)} productos con unidad identificada, al menos "
            f"{MIN_MARKETS} mercados y {MIN_MONTHS} meses. Revise los años o el esquema SIPSA-P."
        )
    best = None
    failures = []
    attempts = 0
    for a, b in itertools.combinations(candidates, 2):
        if norm_text(a["producto"]) == norm_text(b["producto"]):
            continue
        attempts += 1
        chosen = [a, b]
        product_names = [a["producto"], b["producto"]]
        unit_map = {x["producto"]: x["unidad"] for x in chosen}
        variety_map = {x["producto"]: x["variedad"] for x in chosen if norm_text(x["variedad"])}
        try:
            monthly, balanced, markets, dates, summary = prepare_prices(
                raw, product_names,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                unit_per_product=unit_map,
                variety_per_product=variety_map,
                confirm_unitless=False,
                min_markets=MIN_MARKETS,
                min_months=MIN_MONTHS,
            )
            score = int(summary["n_markets"] * summary["n_months"])
            if best is None or score > best["score"]:
                best = {"score": score, "chosen": chosen, "monthly": monthly, "balanced": balanced,
                        "markets": markets, "dates": dates, "summary": summary}
        except Exception as exc:
            if len(failures) < 8:
                failures.append(f"{product_names}: {str(exc).splitlines()[0]}")
    print(f"Variantes candidatas: {len(candidates)}; pares examinados: {attempts}")
    if best is None:
        raise RuntimeError(
            "No encontré dos productos con un panel balanceado suficiente. No se imputaron precios. "
            "Primeros motivos: " + " | ".join(failures)
        )
    # Limita el tamaño del sistema para una primera ejecución reproducible. Entre los
    # mercados que ya tienen cobertura mensual completa, prioriza los que tienen más
    # registros reales en las variantes elegidas (sin completar huecos).
    chosen = best["chosen"]
    chosen_raw = raw.copy()
    chosen_raw["mercado_key"] = chosen_raw["mercado"].map(norm_text)
    chosen_raw["producto_key"] = chosen_raw["producto"].map(norm_text)
    chosen_raw["unidad_key"] = chosen_raw["unidad"].map(norm_text)
    chosen_raw["variedad_key"] = chosen_raw["variedad"].map(norm_text)
    chosen_raw = chosen_raw[chosen_raw["fecha"].between(start_date, end_date)].copy()
    counts = {}
    for item in chosen:
        m = chosen_raw["producto_key"].eq(norm_text(item["producto"]))
        m &= chosen_raw["unidad_key"].eq(norm_text(item["unidad"]))
        m &= chosen_raw["variedad_key"].eq(norm_text(item["variedad"]))
        for market_key, count in chosen_raw.loc[m].groupby("mercado_key").size().items():
            counts[market_key] = counts.get(market_key, 0) + int(count)
    eligible_keys = set(best["balanced"]["mercado_key"].unique())
    ordered_keys = sorted((k for k in eligible_keys if k), key=lambda k: (-counts.get(k, 0), k))
    selected_keys = set(ordered_keys[:MAX_MARKETS])
    if len(selected_keys) < MIN_MARKETS:
        raise RuntimeError(f"Tras limitar el panel quedan menos de {MIN_MARKETS} mercados completos.")
    best["balanced"] = best["balanced"][best["balanced"]["mercado_key"].isin(selected_keys)].copy()
    best["monthly"] = best["monthly"][best["monthly"]["mercado_key"].isin(selected_keys)].copy()
    display_names = (best["balanced"].groupby("mercado_key")["mercado"].first().to_dict())
    market_keys_order = sorted(selected_keys)
    best["markets"] = [display_names[k] for k in market_keys_order]
    best["summary"]["n_markets"] = len(market_keys_order)
    best["summary"]["markets_display"] = best["markets"]
    best["summary"]["markets_key"] = market_keys_order
    best["score"] = int(best["summary"]["n_markets"] * best["summary"]["n_months"])
    return (best["chosen"], best["monthly"], best["balanced"], best["markets"], best["dates"], best["summary"])


def ensure_market_coordinates(market_names: list[str]) -> pd.DataFrame:
    """Obtiene coordenadas de mercado con una jerarquía explícita y caché incremental.

    1) Usa coordenadas conocidas para instalaciones documentadas.
    2) Intenta el nombre/alias de la instalación en Nominatim.
    3) Si no encuentra la instalación, usa una ubicación de ciudad/municipio SOLO como
       aproximación explícitamente marcada. Así no se detiene el flujo por nombres SIPSA
       que no coinciden literalmente con OSM. Esas coordenadas deben verificarse antes de
       cualquier interpretación espacial sustantiva.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if COORDS_FILE.exists():
        existing = pd.read_csv(COORDS_FILE)
    else:
        existing = pd.DataFrame(columns=[
            "mercado", "latitud", "longitud", "nombre_geocodificado",
            "fuente_coordenada", "precision_geografica"
        ])
    required = {"mercado", "latitud", "longitud"}
    if not required.issubset(existing.columns):
        raise RuntimeError(f"{COORDS_FILE} debe tener columnas: mercado, latitud, longitud.")

    existing["mercado_key"] = existing["mercado"].map(norm_text)
    existing = existing.drop_duplicates("mercado_key", keep="last")
    by_key = {row["mercado_key"]: row.to_dict() for _, row in existing.iterrows()}

    def save_cache() -> None:
        rows = []
        for row in by_key.values():
            if isinstance(row, pd.Series):
                row = row.to_dict()
            rows.append(dict(row))
        output = pd.DataFrame(rows).drop(columns=["mercado_key"], errors="ignore")
        # Stable schema, while retaining any useful columns previously saved.
        base_cols = ["mercado", "latitud", "longitud", "nombre_geocodificado",
                     "fuente_coordenada", "precision_geografica"]
        for col in base_cols:
            if col not in output.columns:
                output[col] = "" if col not in {"latitud", "longitud"} else np.nan
        output.to_csv(COORDS_FILE, index=False, encoding="utf-8-sig")

    def known_for_key(key: str) -> dict[str, Any] | None:
        if key in KNOWN_MARKET_COORDINATES:
            return KNOWN_MARKET_COORDINATES[key]
        # Robust aliases where the market name in SIPSA has city/facility order changes.
        if "granabastos" in key or "gran abastos" in key:
            return KNOWN_MARKET_COORDINATES["barranquilla granabastos"]
        if "mercar" in key and "armenia" in key:
            return KNOWN_MARKET_COORDINATES["armenia mercar"]
        return None

    def candidate_queries(name: str) -> tuple[list[str], list[str]]:
        key = norm_text(name)
        exact = [f"{name}, Colombia"]
        if "granabastos" in key or "gran abastos" in key:
            exact += [
                "GranAbastos, Soledad, Atlántico, Colombia",
                "Gran Central de Abastos del Caribe, Soledad, Atlántico, Colombia",
            ]
        if "mercar" in key and "armenia" in key:
            exact += ["Mercar Central Mayorista Armenia, Quindío, Colombia"]
        for token, alias in [
            ("corabastos", "Corabastos, Bogotá, Colombia"),
            ("centroabastos", "Centroabastos, Bucaramanga, Santander, Colombia"),
            ("cenabastos", "Cenabastos, Cúcuta, Norte de Santander, Colombia"),
            ("cavasa", "CAVASA, Candelaria, Valle del Cauca, Colombia"),
            ("mercasa", "Mercasa, Pereira, Risaralda, Colombia"),
            ("surabastos", "Surabastos, Neiva, Huila, Colombia"),
            ("mayorista de antioquia", "Central Mayorista de Antioquia, Itagüí, Colombia"),
            ("potrerillo", "Plaza de Mercado El Potrerillo, Pasto, Nariño, Colombia"),
        ]:
            if token in key:
                exact.append(alias)
        # City/municipality fallback is kept separate and explicitly labeled as approximate.
        parts = [p.strip() for p in re.split(r"[,;|/–—-]+", str(name)) if p.strip()]
        cities = parts[:1]
        known_city_aliases = [
            ("bogota", "Bogotá, Colombia"), ("medellin", "Medellín, Colombia"),
            ("cali", "Cali, Valle del Cauca, Colombia"), ("barranquilla", "Barranquilla, Colombia"),
            ("armenia", "Armenia, Quindío, Colombia"), ("cartagena", "Cartagena de Indias, Bolívar, Colombia"),
            ("bucaramanga", "Bucaramanga, Santander, Colombia"), ("cucuta", "Cúcuta, Norte de Santander, Colombia"),
            ("ibague", "Ibagué, Tolima, Colombia"), ("pasto", "Pasto, Nariño, Colombia"),
            ("neiva", "Neiva, Huila, Colombia"), ("pereira", "Pereira, Risaralda, Colombia"),
            ("villavicencio", "Villavicencio, Meta, Colombia"), ("monteria", "Montería, Córdoba, Colombia"),
            ("sincelejo", "Sincelejo, Sucre, Colombia"), ("popayan", "Popayán, Cauca, Colombia"),
            ("manizales", "Manizales, Caldas, Colombia"), ("tunja", "Tunja, Boyacá, Colombia"),
            ("valledupar", "Valledupar, Cesar, Colombia"), ("santa marta", "Santa Marta, Magdalena, Colombia"),
            ("riohacha", "Riohacha, La Guajira, Colombia"), ("florencia", "Florencia, Caquetá, Colombia"),
            ("yopal", "Yopal, Casanare, Colombia"), ("quibdo", "Quibdó, Chocó, Colombia"),
            ("soledad", "Soledad, Atlántico, Colombia"), ("barrancabermeja", "Barrancabermeja, Santander, Colombia"),
        ]
        normalized = key
        city_queries = []
        for token, query in known_city_aliases:
            if token in normalized:
                city_queries.append(query)
        if cities:
            city_queries.insert(0, f"{cities[0]}, Colombia")
        # Dedupe while preserving order and remove exact queries from fallback list.
        exact = list(dict.fromkeys(exact))
        city_queries = [q for q in dict.fromkeys(city_queries) if q not in exact]
        return exact, city_queries

    # Load geocoder only if there are names that don't have cached/known coordinates.
    missing: list[str] = []
    for name in market_names:
        key = norm_text(name)
        known = known_for_key(key)
        if known is not None:
            by_key[key] = {"mercado": name, **known, "mercado_key": key}
            save_cache()
            continue
        row = by_key.get(key)
        if row is not None and pd.notna(row.get("latitud")) and pd.notna(row.get("longitud")):
            continue
        missing.append(name)

    geocode = None
    if missing:
        try:
            from geopy.geocoders import Nominatim
            from geopy.extra.rate_limiter import RateLimiter
        except ImportError as exc:
            raise RuntimeError("Falta geopy, requerido para geocodificar mercados nuevos.") from exc
        geolocator = Nominatim(user_agent="agroclima-static-colombia/1.1 (research dashboard; OSM geocoding)", timeout=30)
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.1, max_retries=2,
                              error_wait_seconds=3, swallow_exceptions=True)

    for name in missing:
        key = norm_text(name)
        exact_queries, city_queries = candidate_queries(name)
        found = None
        used_query = None
        for query in exact_queries:
            found = geocode(query, country_codes="co", exactly_one=True)
            if found is not None:
                used_query = query
                break
        precision = "instalación/lugar devuelto por Nominatim; verificar"
        source = "OpenStreetMap Nominatim"
        if found is None:
            for query in city_queries:
                found = geocode(query, country_codes="co", exactly_one=True)
                if found is not None:
                    used_query = query
                    precision = "aproximación a centro urbano/municipal; NO es ubicación exacta del mercado"
                    source = "OpenStreetMap Nominatim; geocodificación fallback municipal, verificar"
                    break
        if found is None:
            # Preserve all coordinates already resolved so that a future attempt doesn't start over.
            save_cache()
            raise RuntimeError(
                f"Nominatim no encontró ni el mercado ni su ciudad para {name!r}. "
                f"Las coordenadas ya encontradas se guardaron en {COORDS_FILE}. "
                "Añada manualmente una coordenada verificada para este mercado y vuelva a ejecutar."
            )
        by_key[key] = {
            "mercado": name,
            "latitud": float(found.latitude),
            "longitud": float(found.longitude),
            "nombre_geocodificado": str(found.address),
            "fuente_coordenada": f"{source}; consulta={used_query}",
            "precision_geografica": precision,
            "mercado_key": key,
        }
        save_cache()  # Incremental: persist even if a later market fails.
        if "NO es ubicación exacta" in precision:
            print(f"ADVERTENCIA geográfica: {name!r} se ubicó usando {used_query!r}; revisar en market_coordinates.csv.")

    output = pd.read_csv(COORDS_FILE) if COORDS_FILE.exists() else pd.DataFrame(list(by_key.values()))
    output["mercado_key"] = output["mercado"].map(norm_text)
    result = output[output["mercado_key"].isin({norm_text(x) for x in market_names})].copy()
    result["latitud"] = pd.to_numeric(result["latitud"], errors="coerce")
    result["longitud"] = pd.to_numeric(result["longitud"], errors="coerce")
    result = result.drop_duplicates("mercado_key", keep="last")
    if len(result) != len(set(norm_text(x) for x in market_names)) or result[["latitud", "longitud"]].isna().any().any():
        raise RuntimeError(f"{COORDS_FILE} no cubre todos los mercados o contiene coordenadas vacías.")
    if not result["latitud"].between(-5.5, 14.5).all() or not result["longitud"].between(-83.5, -65.0).all():
        raise RuntimeError("Hay coordenadas fuera del rango territorial aproximado de Colombia. Revise market_coordinates.csv.")
    return result.drop(columns=["mercado_key"], errors="ignore").reset_index(drop=True)


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    clean = df.copy()
    for col in clean.columns:
        if pd.api.types.is_datetime64_any_dtype(clean[col]):
            clean[col] = clean[col].map(lambda x: x.isoformat() if pd.notna(x) else None)
    return json_safe(clean.replace({np.nan: None}).to_dict(orient="records"))


def download_department_boundaries() -> None:
    """Guarda los límites departamentales MGN 2025 usados por el mapa coroplético."""
    def validate(payload: Any) -> list[dict[str, Any]]:
        if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
            raise RuntimeError("La cartografía departamental no es una FeatureCollection GeoJSON.")
        features = payload.get("features", [])
        if len(features) != 33:
            raise RuntimeError("La cartografía departamental no contiene las 33 entidades MGN esperadas.")
        for feature in features:
            props = feature.get("properties", {})
            if not feature.get("geometry") or not props.get("divipola_dep") or not props.get("dpto_nombre"):
                raise RuntimeError("La cartografía departamental no contiene geometrías y campos DIVIPOLA esperados.")
        return features

    try:
        response = requests.get(DEPARTMENTS_GEOJSON_URL, timeout=(15, 90))
        response.raise_for_status()
        payload = response.json()
        features = validate(payload)
    except (requests.RequestException, ValueError, RuntimeError) as exc:
        if DEPARTMENTS_GEOJSON.is_file():
            cached = json.loads(DEPARTMENTS_GEOJSON.read_text(encoding="utf-8"))
            validate(cached)
            print(f"Aviso: no se pudo actualizar la cartografía ({exc}); se conserva la copia MGN ya guardada.")
            return
        raise RuntimeError(f"No se pudo descargar la cartografía departamental MGN: {exc}") from exc
    payload["source"] = "DANE, Marco Geoestadístico Nacional 2025; geometría simplificada del repositorio mapas-colombia (CC BY 4.0)."
    temp = DEPARTMENTS_GEOJSON.with_suffix(".geojson.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    temp.replace(DEPARTMENTS_GEOJSON)
    print(f"Cartografía departamental lista: {DEPARTMENTS_GEOJSON} ({DEPARTMENTS_GEOJSON.stat().st_size:,} bytes).")


def build_residual_panel(results: dict[str, Any], coords: pd.DataFrame) -> pd.DataFrame:
    """Expone residuos de ajuste por producto, mercado y mes para el mapa departamental."""
    coordinate_rows = coords.copy()
    coordinate_rows["mercado_key"] = coordinate_rows["mercado"].map(norm_text)
    coordinate_rows = coordinate_rows[["mercado_key", "latitud", "longitud"]].drop_duplicates("mercado_key")
    output = []
    for product, fit in results.items():
        temp = fit.resids.rename("residuo").reset_index()
        if temp.shape[1] < 3:
            raise ValueError(f"No se pudieron identificar mercado y fecha en los residuos de {product!r}.")
        temp = temp.rename(columns={temp.columns[0]: "mercado_key", temp.columns[1]: "fecha_residuo"})
        temp["mercado_key"] = temp["mercado_key"].map(norm_text)
        temp["fecha_residuo"] = pd.to_datetime(temp["fecha_residuo"], errors="coerce")
        temp["residuo"] = pd.to_numeric(temp["residuo"], errors="coerce")
        temp = temp.dropna(subset=["fecha_residuo", "residuo"])
        temp = temp.merge(coordinate_rows, on="mercado_key", how="left", validate="many_to_one")
        if temp[["latitud", "longitud"]].isna().any().any():
            missing = sorted(temp.loc[temp["latitud"].isna(), "mercado_key"].unique())
            raise ValueError(f"Faltan coordenadas para residuos de los mercados: {missing}")
        temp["producto"] = str(product)
        output.append(temp[["producto", "mercado_key", "fecha_residuo", "residuo", "latitud", "longitud"]])
    if not output:
        return pd.DataFrame(columns=["producto", "mercado_key", "fecha_residuo", "residuo", "latitud", "longitud"])
    return pd.concat(output, ignore_index=True).sort_values(["producto", "fecha_residuo", "mercado_key"]).reset_index(drop=True)


def main() -> None:
    if START_YEAR < 2013 or END_YEAR > 2024 or START_YEAR > END_YEAR:
        raise ValueError("El catálogo enlazado actualmente cubre SIPSA-P 2013–2024; revise START_YEAR/END_YEAR.")
    if MIN_MARKETS < 4 or MIN_MONTHS < 24:
        raise ValueError("Por seguridad econométrica, use como mínimo 4 mercados y 24 meses.")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    download_department_boundaries()
    force_refresh = os.getenv("AGROCLIMA_FORCE_REFRESH", "1").strip().lower() in {"1", "true", "yes", "si"}
    print(f"Descargando SIPSA-P real de {START_YEAR} a {END_YEAR} desde DANE (forzar actualización={force_refresh})...")
    if force_refresh and RAW_DIR.exists():
        # Si cambia un ZIP, obliga también a extraerlo de nuevo (no reutilizar marcadores viejos).
        for extracted in RAW_DIR.rglob("*__extracted"):
            if extracted.is_dir():
                shutil.rmtree(extracted, ignore_errors=True)
    archives = download_sipsa_archives(START_YEAR, END_YEAR, RAW_DIR, force=force_refresh)
    sipsa_scale_min = float(os.getenv("AGROCLIMA_SIPSA_AUDIT_MIN_SCALE_RATIO", "0.1"))
    sipsa_scale_max = float(os.getenv("AGROCLIMA_SIPSA_AUDIT_MAX_SCALE_RATIO", "10"))
    sipsa_audit_collector = SipsaAuditCollector(RAW_DIR, START_YEAR, END_YEAR, sipsa_scale_min, sipsa_scale_max)
    raw, audit = load_sipsa_data(RAW_DIR, table_audit_callback=sipsa_audit_collector.capture_table)
    raw["fecha"] = pd.to_datetime(raw["fecha"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    raw = raw[raw["fecha"].between(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31")].copy()
    # Auditoría explícita de qué archivos y años logró leer el parser SIPSA.
    if not audit.empty:
        print("[SIPSA audit] Archivos/hojas reconocidos (primeras 30 filas):")
        audit_cols = [c for c in ["archivo_hoja", "filas_canonicas", "fecha_min", "fecha_max", "columnas_detectadas"] if c in audit.columns]
        print(audit[audit_cols].head(30).to_string(index=False))
    if not raw.empty:
        by_year = raw["fecha"].dt.year.value_counts().sort_index().to_dict()
        print(f"[SIPSA audit] Cobertura canónica total: {raw['fecha'].min():%Y-%m} a {raw['fecha'].max():%Y-%m}; observaciones/año={by_year}")
        if raw["fecha"].max().year < END_YEAR:
            print(f"ADVERTENCIA: se pidió hasta {END_YEAR}, pero las tablas reconocidas solo llegan a {raw['fecha'].max():%Y-%m}. Revisar el audit por archivo; no asumir que los años posteriores se han leído correctamente.")
    # Normaliza cadenas que el lector puede haber convertido desde celdas vacías a "nan".
    for col in ["mercado", "producto", "unidad", "variedad"]:
        if col in raw.columns:
            raw[col] = raw[col].fillna("").astype(str).str.strip()
            raw.loc[raw[col].map(norm_text).isin(["", "nan", "none", "null", "na"]), col] = ""
    raw["mercado_key"] = raw["mercado"].map(norm_text)
    raw["producto_key"] = raw["producto"].map(norm_text)
    raw["unidad_key"] = raw["unidad"].map(norm_text)
    raw["variedad_key"] = raw["variedad"].map(norm_text)
    if raw.empty:
        raise RuntimeError("Los ZIP oficiales se descargaron, pero no se reconocieron observaciones en el rango solicitado.")

    # Genera trazabilidad de las fuentes antes de seleccionar productos/modelar.
    # El generador reconcilia canónicas antes/después de deduplicar; si no cuadra, falla.
    sipsa_audit_paths = sipsa_audit_collector.write(canonical_raw=raw, output_dir=DATA_DIR)
    print("[SIPSA audit] Trazabilidad detallada generada correctamente.")

    # A previous parser could silently discard DANE rows whose dates were stored as
    # Spanish labels like "ene-24". Do not publish a newer JSON with a truncated range.
    year_counts = raw["fecha"].dt.year.value_counts().to_dict()
    missing_years = [year for year in range(START_YEAR, END_YEAR + 1) if not year_counts.get(year, 0)]
    latest_observed_month = raw["fecha"].max().to_period("M").to_timestamp()
    required_latest_month = pd.Timestamp(f"{END_YEAR}-12-01")
    if missing_years or latest_observed_month < required_latest_month:
        count_summary = {year: int(year_counts.get(year, 0)) for year in range(START_YEAR, END_YEAR + 1)}
        raise RuntimeError(
            "Cobertura SIPSA-P incompleta después de leer los archivos oficiales. "
            f"Conteo de filas por año: {count_summary}; último mes reconocido: "
            f"{latest_observed_month:%Y-%m}; se esperaba al menos {required_latest_month:%Y-%m}. "
            "Se cancela la actualización para no publicar un JSON truncado. "
            "Revise las filas [SIPSA audit] y la interpretación de la columna Fecha."
        )

    start = pd.Timestamp(f"{START_YEAR}-01-01")
    raw_end = raw["fecha"].max().to_period("M").to_timestamp()
    configured_end = pd.Timestamp(f"{END_YEAR}-12-01")
    end = min(raw_end, configured_end)
    chosen, monthly, balanced_prices, market_names, dates, summary = choose_best_pair(raw, start, end)
    product_names = [x["producto"] for x in chosen]
    assert_selected_products_scale(
        sipsa_audit_paths["year_product_csv"],
        product_names,
        min_ratio=sipsa_scale_min,
        max_ratio=sipsa_scale_max,
    )
    print(f"[SIPSA audit] Control interanual de escala superado para los productos seleccionados ({sipsa_scale_min:g}–{sipsa_scale_max:g}).")
    unit_map = {x["producto"]: x["unidad"] for x in chosen}
    variety_map = {x["producto"]: x["variedad"] for x in chosen if norm_text(x["variedad"])}

    print(f"Productos seleccionados por cobertura conjunta: {product_names}")
    print(f"Panel de precios: {summary['n_markets']} mercados, {summary['n_months']} meses, {summary['start']} a {summary['end']}.")
    coords = ensure_market_coordinates(market_names)
    w, W_dense, markets_keys, coords_sorted, w_meta = build_knn_weights(coords, k=KNN_K)
    dates = pd.DatetimeIndex(dates)
    climate = fetch_nasa_power(coords_sorted, str(dates.min().date()), str(dates.max().date()), CLIMATE_CACHE)
    panel, panel_meta = build_spvarx_panel(balanced_prices, climate, markets_keys, dates, product_names, W_dense)
    fits, coef_df, model_summary = estimate_spvarx(panel, product_names, bandwidth=min(4, max(1, panel["fecha"].nunique() - 1)))
    try:
        model_comparison = compare_spatial_model_families(
            panel, product_names, markets_keys, W_dense, fits
        )
        available_comparators = sum(
            row.get("estado", "").startswith("Estimado")
            for row in model_comparison if row.get("modelo") != "SpVARX"
        )
        print(f"[Modelos espaciales auxiliares] Estimaciones disponibles: {available_comparators}/{3 * len(product_names)}.")
    except Exception as exc:
        model_comparison = []
        print(f"[Modelos espaciales auxiliares] No se pudo completar la comparación: {exc}")
    stability = model_stability(fits, product_names, W_dense)
    print(f"[IRF bootstrap] Calculando bandas con {IRF_BOOTSTRAP_REPS} réplicas...")
    irfs = compute_irfs(
        fits, product_names, W_dense, horizon=IRF_HORIZON, panel=panel,
        bootstrap_repetitions=IRF_BOOTSTRAP_REPS, bootstrap_seed=IRF_BOOTSTRAP_SEED,
    )
    irf_bootstrap_valid = int(irfs["n_bootstrap_valid"].max()) if "n_bootstrap_valid" in irfs and not irfs.empty else 0
    irf_bootstrap_available = bool(irfs["bootstrap_interval_available"].all()) if "bootstrap_interval_available" in irfs and not irfs.empty else False
    irf_bootstrap_block_months = int(irfs["bootstrap_block_months"].max()) if "bootstrap_block_months" in irfs and not irfs.empty else None
    print(f"[IRF bootstrap] Réplicas estimables: {irf_bootstrap_valid}/{IRF_BOOTSTRAP_REPS}; bandas disponibles: {irf_bootstrap_available}.")
    print(f"[Robustez espacial] Reestimando con k={','.join(map(str, KNN_SENSITIVITY_VALUES))} y la misma muestra...")
    spatial_robustness = spatial_weights_sensitivity(
        panel, coords_sorted, product_names, k_values=KNN_SENSITIVITY_VALUES,
        base_k=KNN_K, base_weights=W_dense, base_results=fits,
        base_coefficients=coef_df, base_irfs=irfs, horizon=IRF_HORIZON,
        bootstrap_repetitions=IRF_BOOTSTRAP_REPS, bootstrap_seed=IRF_BOOTSTRAP_SEED,
    )
    available_specs = int((spatial_robustness["summary"].get("estado", pd.Series(dtype=str)) == "Estimación disponible").sum())
    print(f"[Robustez espacial] Especificaciones disponibles: {available_specs}/{len(KNN_SENSITIVITY_VALUES)}.")
    adf = stationarity_audit(panel, product_names)
    try:
        moran = residual_moran(fits, w, markets_keys)
    except Exception as exc:
        moran = pd.DataFrame([{"advertencia": f"No fue posible calcular Moran de residuos: {exc}"}])

    residual_panel = build_residual_panel(fits, coords_sorted)
    backtest_error = None
    try:
        backtest = rolling_spvarx_backtest(
            balanced_prices, climate, markets_keys, dates, product_names, W_dense,
            min_train_months=BACKTEST_MIN_TRAIN_MONTHS,
        )
    except Exception as exc:
        backtest = pd.DataFrame(columns=[
            "producto", "mercado_key", "fecha", "latitud", "longitud", "observado",
            "prediccion_spvarx", "prediccion_sin_cambio", "prediccion_persistencia",
            "meses_entrenamiento",
        ])
        backtest_error = str(exc)
        print(f"[Backtest] No se pudo completar la validación temporal: {exc}")
    backtest_temp = BACKTEST_CSV.with_suffix(".csv.tmp")
    backtest.to_csv(backtest_temp, index=False, encoding="utf-8-sig")
    backtest_temp.replace(BACKTEST_CSV)
    if not backtest.empty:
        print(f"[Backtest] {len(backtest):,} predicciones retenidas para validación temporal.")

    # Price series shown in the frontend are actual DANE market observations, summarized by month/product.
    price_series = (balanced_prices.groupby(["fecha", "producto"], as_index=False)
                    .agg(precio_promedio_mercados=("precio", "mean"), n_mercados=("mercado_key", "nunique")))
    latest_date = panel["fecha"].max()
    latest_panel = panel[panel["fecha"] == latest_date].copy()
    latest_panel = latest_panel.set_index("mercado_key")
    market_payload = []
    coord_work = coords_sorted.copy()
    coord_work["mercado_key"] = coord_work["mercado"].map(norm_text)
    for _, c in coord_work.iterrows():
        key = c["mercado_key"]
        r = latest_panel.loc[key] if key in latest_panel.index else None
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        changes = {}
        if r is not None:
            for j, product in enumerate(product_names):
                changes[product] = float(r.get(f"y{j}", np.nan))
        market_payload.append({
            "mercado": str(c["mercado"]), "latitud": float(c["latitud"]), "longitud": float(c["longitud"]),
            "nombre_geocodificado": str(c.get("nombre_geocodificado", "")),
            "fuente_coordenada": str(c.get("fuente_coordenada", "")),
            "precision_geografica": str(c.get("precision_geografica", "")),
            "cambios_log_precio": changes,
        })

    stability_payload = {
        "spectral_radius_transition": float(stability["spectral_radius"]),
        "stable": bool(stability["stable"]),
        "interpretation": "Estabilidad de la transición temporal-espacial rezagada estimada; no es evidencia causal.",
    }
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "start_date": pd.Timestamp(summary["start"]).strftime("%Y-%m"),
        "end_date": pd.Timestamp(summary["end"]).strftime("%Y-%m"),
        "n_markets": int(summary["n_markets"]), "n_months": int(summary["n_months"]),
        "n_model_observations": int(panel_meta["n_obs"]),
        "products": [{"name": p["producto"], "unit": p["unidad"], "variety": p["variedad"]} for p in chosen],
        "selection_rule": f"Dos productos con mayor cobertura conjunta entre hasta {MAX_CANDIDATE_PRODUCTS} candidatos; mínimo {MIN_MARKETS} mercados y {MIN_MONTHS} meses completos.",
        "weights": json_safe(w_meta), "stability": stability_payload,
        "sources": {
            "prices": DANE_CATALOG_URL,
            "climate": "https://power.larc.nasa.gov/docs/services/api/temporal/monthly/",
            "geocoding": "https://nominatim.openstreetmap.org/",
        },
        "source_archive_count": len(archives),
        "source_audit_tables_recognized": int((audit.get("filas_canonicas", pd.Series(dtype=float)).fillna(0) > 0).sum()) if not audit.empty else 0,
        "data_note": "Precios nominales mensuales observados de SIPSA-P. NASA POWER aporta variables meteorológicas derivadas de datos/modelos de NASA; no son mediciones directas de estación en cada mercado.",
        "method_note": "Panel SpVARX(1) reducido estimado ecuación por ecuación con efectos de mercado y tiempo. Incluye W*y[t-1], no W*y[t]. La inferencia con efectos fijos y rezagos endógenos requiere contrastes adicionales.",
        "model_comparison_note": "Comparación complementaria por ecuación y producto entre el SpVARX principal y paneles gaussianos SAR, SEM y SAC estimados en la misma muestra y con efectos fijos de mercado y tiempo. El AIC menor indica mejor ajuste relativo en muestra bajo estos supuestos; no equivale a mejor pronóstico ni a evidencia causal. Las especificaciones auxiliares son univariadas y sus parámetros espaciales no son directamente equivalentes a los coeficientes dinámicos W*y[t-1] del SpVARX.",
        "irf_note": (
            "IRF de un choque climático de +1 DE con intervalos percentiles aproximados del 95% por horizonte, calculados mediante bootstrap circular de bloques temporales completos. Son intervalos punto a punto, no bandas simultáneas para toda la curva. Condicionan en la matriz W, mercados, productos y regresores observados; no corrigen el posible sesgo del panel dinámico ni identifican causalidad. Si un intervalo contiene cero, esa respuesta no se distingue de cero al nivel descriptivo del intervalo."
            if irf_bootstrap_available else
            f"No se alcanzó el mínimo de réplicas válidas para formar bandas IRF ({irf_bootstrap_valid}/{IRF_BOOTSTRAP_REPS}). Se muestran respuestas puntuales; no deben interpretarse causalmente."
        ),
        "irf_bootstrap_method": "Bootstrap circular móvil por bloques de meses completos; cada réplica reestima el SpVARX con efectos fijos de mercado y tiempo. Intervalos percentiles punto a punto por horizonte, 2,5–97,5%.",
        "irf_bootstrap_requested": IRF_BOOTSTRAP_REPS,
        "irf_bootstrap_valid": irf_bootstrap_valid,
        "irf_bootstrap_block_months": irf_bootstrap_block_months,
        "irf_bootstrap_intervals_available": irf_bootstrap_available,
        "spatial_weights_note": f"Análisis de sensibilidad del SpVARX con k={', '.join(map(str, KNN_SENSITIVITY_VALUES))}, usando la misma muestra de meses, mercados y productos. La matriz kNN se simetriza y estandariza por filas; por ello el grado final puede exceder k. Cada alternativa reestima los coeficientes y respuestas impulso, con intervalos bootstrap puntuales cuando hay réplicas suficientes. Los intervalos de cada alternativa son condicionales a su propia W; no son una prueba formal de igualdad entre matrices ni una selección automática del mejor k.",
        "spatial_weights_k_values": list(KNN_SENSITIVITY_VALUES),
        "coordinate_note": "Las coordenadas se guardan en data/market_coordinates.csv. Las coordenadas de instalación documentadas y los fallback al centro urbano llevan fuente y precisión explícitas; el fallback municipal no equivale a la ubicación exacta del mercado y debe revisarse antes de interpretar relaciones espaciales.",
        "residual_note": "Los residuos del mapa son residuos de ajuste en muestra del SpVARX sobre las mismas observaciones usadas en la estimación. El sesgo medio se centra en cero; MAE y RMSE son no negativos y menor indica menor error descriptivo. No son métricas de predicción fuera de muestra. La respuesta está en unidades de Δlog(precio).",
        "backtest_note": "Validación temporal de un paso con origen móvil. Cada corte reestima una ecuación predictiva con efectos fijos de mercado y sin efectos fijos de tiempo, porque el efecto del mes futuro no se conoce al pronosticar. El clima contemporáneo futuro se fija en anomalía cero (escenario normal) y el clima rezagado usa el último mes observado. Es una proyección condicional a clima normal; no es un pronóstico meteorológico ni una prueba causal. La pareja de productos se seleccionó por cobertura en el periodo completo y se mantiene fija, por lo que el backtest evalúa esa pareja ya definida.",
        "backtest_min_train_months": BACKTEST_MIN_TRAIN_MONTHS,
        "backtest_error": backtest_error,
        "department_boundaries_source": "DANE MGN 2025; geometría simplificada: https://github.com/ytolosa/mapas-colombia (CC BY 4.0).",
        "diagnostic_summary": {
            "adf_reject_share": float(adf["rechaza_raiz_unitaria_5pct"].mean()) if "rechaza_raiz_unitaria_5pct" in adf.columns and len(adf) else None,
            "moran_residuals": records(moran),
            "stable_transition": bool(stability["stable"]),
        },
    }
    payload = {
        "status": "ready", "generated_at": meta["generated_at"],
        "message": "Datos oficiales procesados; revise las notas metodológicas antes de interpretar los resultados.",
        "metadata": meta,
        "price_series": records(price_series),
        "markets": json_safe(market_payload),
        "residual_panel": records(residual_panel),
        "rolling_backtest": records(backtest),
        "coefficients": records(coef_df),
        "irfs": records(irfs),
        "model_comparison": json_safe(model_comparison),
        "spatial_robustness": {
            "summary": records(spatial_robustness["summary"]),
            "coefficients": records(spatial_robustness["coefficients"]),
            "irfs": records(spatial_robustness["irfs"]),
        },
        "diagnostics": {"adf": records(adf), "moran": records(moran)},
    }
    # Strict JSON and atomic replacement: a failed pipeline never corrupts the last publication.
    text = json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False)
    temp = OUTPUT_JSON.with_suffix(".json.tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(OUTPUT_JSON)
    print(f"JSON listo: {OUTPUT_JSON} ({OUTPUT_JSON.stat().st_size:,} bytes)")
    print(f"Estabilidad de transición: {stability['stable']} (radio={stability['spectral_radius']:.4f})")
    print("Advertencia: las coordenadas autogeocodificadas requieren inspección sustantiva.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: la actualización falló: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
