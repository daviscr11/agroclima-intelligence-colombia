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
import sys
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dane_auto import DANE_CATALOG_URL, download_sipsa_archives
from agroclima_spvarx_real import (
    norm_text,
    load_sipsa_data,
    prepare_prices,
    fetch_nasa_power,
    build_knn_weights,
    build_spvarx_panel,
    estimate_spvarx,
    model_stability,
    compute_irfs,
    stationarity_audit,
    residual_moran,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RUNTIME_DIR = ROOT / ".runtime"
RAW_DIR = RUNTIME_DIR / "sipsa_raw"
CLIMATE_CACHE = RUNTIME_DIR / "nasa_power_monthly.csv"
COORDS_FILE = DATA_DIR / "market_coordinates.csv"
OUTPUT_JSON = DATA_DIR / "observatorio.json"

START_YEAR = int(os.getenv("AGROCLIMA_START_YEAR", "2019"))
END_YEAR = int(os.getenv("AGROCLIMA_END_YEAR", "2024"))
MIN_MARKETS = int(os.getenv("AGROCLIMA_MIN_MARKETS", "6"))
MIN_MONTHS = int(os.getenv("AGROCLIMA_MIN_MONTHS", "36"))
MAX_CANDIDATE_PRODUCTS = int(os.getenv("AGROCLIMA_MAX_CANDIDATE_PRODUCTS", "24"))
MAX_MARKETS = int(os.getenv("AGROCLIMA_MAX_MARKETS", "12"))
KNN_K = int(os.getenv("AGROCLIMA_KNN_K", "4"))
IRF_HORIZON = int(os.getenv("AGROCLIMA_IRF_HORIZON", "18"))


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
    """Conserva coordenadas cacheadas; geocodifica solo los mercados nuevos.

    Nominatim es una fuente auxiliar de geocodificación, no una fuente de precios/clima.
    Se registran los resultados para poder revisarlos y evitar geocodificar cada mes.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if COORDS_FILE.exists():
        existing = pd.read_csv(COORDS_FILE)
    else:
        existing = pd.DataFrame(columns=["mercado", "latitud", "longitud", "nombre_geocodificado", "fuente_coordenada"])
    required = {"mercado", "latitud", "longitud"}
    if not required.issubset(existing.columns):
        raise RuntimeError(f"{COORDS_FILE} debe tener columnas: mercado, latitud, longitud.")
    existing["mercado_key"] = existing["mercado"].map(norm_text)
    existing = existing.drop_duplicates("mercado_key", keep="last")
    missing = []
    by_key = {row["mercado_key"]: row for _, row in existing.iterrows()}
    for name in market_names:
        key = norm_text(name)
        row = by_key.get(key)
        if row is None or pd.isna(row.get("latitud")) or pd.isna(row.get("longitud")):
            missing.append(name)
    if missing:
        try:
            from geopy.geocoders import Nominatim
            from geopy.extra.rate_limiter import RateLimiter
        except ImportError as exc:
            raise RuntimeError("Falta geopy, requerido para geocodificar mercados nuevos.") from exc
        geolocator = Nominatim(user_agent="agroclima-static-colombia/1.0 (research dashboard)", timeout=30)
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.1, max_retries=2,
                              error_wait_seconds=3, swallow_exceptions=True)
        for name in missing:
            found = geocode(f"{name}, Colombia", country_codes="co", exactly_one=True)
            if found is None:
                raise RuntimeError(
                    f"Nominatim no encontró coordenadas para el mercado {name!r}. "
                    f"Complete manualmente {COORDS_FILE} con latitud/longitud y vuelva a ejecutar."
                )
            by_key[norm_text(name)] = {
                "mercado": name, "latitud": float(found.latitude), "longitud": float(found.longitude),
                "nombre_geocodificado": str(found.address), "fuente_coordenada": "OpenStreetMap Nominatim; revisar",
                "mercado_key": norm_text(name),
            }
    output = pd.DataFrame(list(by_key.values()))
    output = output.drop(columns=["mercado_key"], errors="ignore")
    output.to_csv(COORDS_FILE, index=False, encoding="utf-8-sig")
    result = output.copy()
    result["mercado_key"] = result["mercado"].map(norm_text)
    result = result[result["mercado_key"].isin({norm_text(x) for x in market_names})].copy()
    result["latitud"] = pd.to_numeric(result["latitud"], errors="coerce")
    result["longitud"] = pd.to_numeric(result["longitud"], errors="coerce")
    if len(result) != len(set(norm_text(x) for x in market_names)) or result[["latitud", "longitud"]].isna().any().any():
        raise RuntimeError("La tabla de coordenadas no cubre todos los mercados o contiene coordenadas vacías.")
    # Broad sanity bounds; this is not proof that a market was geocoded to the right facility.
    if not result["latitud"].between(-5.5, 14.5).all() or not result["longitud"].between(-83.5, -65.0).all():
        raise RuntimeError("Hay coordenadas fuera del rango territorial aproximado de Colombia. Revise market_coordinates.csv.")
    return result.drop_duplicates("mercado_key").reset_index(drop=True)


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    clean = df.copy()
    for col in clean.columns:
        if pd.api.types.is_datetime64_any_dtype(clean[col]):
            clean[col] = clean[col].map(lambda x: x.isoformat() if pd.notna(x) else None)
    return json_safe(clean.replace({np.nan: None}).to_dict(orient="records"))


def main() -> None:
    if START_YEAR < 2013 or END_YEAR > 2024 or START_YEAR > END_YEAR:
        raise ValueError("El catálogo enlazado actualmente cubre SIPSA-P 2013–2024; revise START_YEAR/END_YEAR.")
    if MIN_MARKETS < 4 or MIN_MONTHS < 24:
        raise ValueError("Por seguridad econométrica, use como mínimo 4 mercados y 24 meses.")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    force_refresh = os.getenv("AGROCLIMA_FORCE_REFRESH", "1").strip().lower() in {"1", "true", "yes", "si"}
    print(f"Descargando SIPSA-P real de {START_YEAR} a {END_YEAR} desde DANE (forzar actualización={force_refresh})...")
    if force_refresh and RAW_DIR.exists():
        # Si cambia un ZIP, obliga también a extraerlo de nuevo (no reutilizar marcadores viejos).
        for extracted in RAW_DIR.rglob("*__extracted"):
            if extracted.is_dir():
                shutil.rmtree(extracted, ignore_errors=True)
    archives = download_sipsa_archives(START_YEAR, END_YEAR, RAW_DIR, force=force_refresh)
    raw, audit = load_sipsa_data(RAW_DIR)
    raw["fecha"] = pd.to_datetime(raw["fecha"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    raw = raw[raw["fecha"].between(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31")].copy()
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

    start = pd.Timestamp(f"{START_YEAR}-01-01")
    raw_end = raw["fecha"].max().to_period("M").to_timestamp()
    configured_end = pd.Timestamp(f"{END_YEAR}-12-01")
    end = min(raw_end, configured_end)
    chosen, monthly, balanced_prices, market_names, dates, summary = choose_best_pair(raw, start, end)
    product_names = [x["producto"] for x in chosen]
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
    stability = model_stability(fits, product_names, W_dense)
    irfs = compute_irfs(fits, product_names, W_dense, horizon=IRF_HORIZON)
    adf = stationarity_audit(panel, product_names)
    try:
        moran = residual_moran(fits, w, markets_keys)
    except Exception as exc:
        moran = pd.DataFrame([{"advertencia": f"No fue posible calcular Moran de residuos: {exc}"}])

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
        "irf_note": "IRF puntuales de un choque exógeno +1 DE; no incluyen bandas de confianza ni identificación estructural. No interpretar causalmente.",
        "coordinate_note": "Las coordenadas nuevas se sugieren con Nominatim y se guardan en data/market_coordinates.csv; deben revisarse por posible ambigüedad de nombres de mercado.",
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
        "coefficients": records(coef_df),
        "irfs": records(irfs),
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
