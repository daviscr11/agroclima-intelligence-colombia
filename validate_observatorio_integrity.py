# -*- coding: utf-8 -*-
"""Validador de integridad del JSON del observatorio antes de publicarlo.

Solo usa la biblioteca estándar de Python. Devuelve código 1 si detecta
inconsistencias críticas y así bloquea el commit/publicación del workflow.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any


def month_range(start: str, end: str) -> list[str]:
    """Devuelve meses YYYY-MM inclusivos; valida que las entradas sean válidas."""
    try:
        sy, sm = map(int, start.split("-"))
        ey, em = map(int, end.split("-"))
        date(sy, sm, 1)
        date(ey, em, 1)
    except (TypeError, ValueError) as exc:
        raise ValueError("Las fechas deben tener formato YYYY-MM válido.") from exc
    if (ey, em) < (sy, sm):
        raise ValueError("El mes final no puede ser anterior al inicial.")
    months: list[str] = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y += 1
            m = 1
    return months


def _strict_json_constant(value: str) -> None:
    raise ValueError(f"Constante JSON no válida: {value} (NaN/Infinity no permitidos).")


def load_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream, parse_constant=_strict_json_constant)
    if not isinstance(payload, dict):
        raise ValueError("La raíz del JSON debe ser un objeto.")
    return payload


def collect_integrity_errors(
    payload: dict[str, Any],
    expected_start: str = "2019-01",
    expected_end: str = "2024-12",
    min_monthly_ratio: float = 0.1,
    max_monthly_ratio: float = 10.0,
) -> list[str]:
    """Devuelve fallos críticos de integridad encontrados en el payload."""
    errors: list[str] = []
    expected_months = month_range(expected_start, expected_end)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return ["Falta el objeto metadata o no es válido."]

    if payload.get("status") != "ready":
        errors.append(f"status debe ser 'ready'; recibido: {payload.get('status')!r}.")
    if metadata.get("start_date") != expected_start:
        errors.append(f"metadata.start_date={metadata.get('start_date')!r}; se esperaba {expected_start!r}.")
    if metadata.get("end_date") != expected_end:
        errors.append(f"metadata.end_date={metadata.get('end_date')!r}; se esperaba {expected_end!r}.")
    if metadata.get("n_months") != len(expected_months):
        errors.append(f"metadata.n_months={metadata.get('n_months')!r}; se esperaban {len(expected_months)} meses.")

    products_meta = metadata.get("products")
    if not isinstance(products_meta, list) or not products_meta:
        errors.append("metadata.products debe ser una lista no vacía.")
        products: list[str] = []
    else:
        products = [p.get("name") for p in products_meta if isinstance(p, dict) and isinstance(p.get("name"), str) and p.get("name").strip()]
        if len(products) != len(products_meta):
            errors.append("Cada producto en metadata.products debe tener un nombre no vacío.")
        if len(set(products)) != len(products):
            errors.append("Hay productos duplicados en metadata.products.")

    try:
        n_markets = int(metadata.get("n_markets"))
        if n_markets < 1:
            raise ValueError
    except (TypeError, ValueError):
        n_markets = 0
        errors.append("metadata.n_markets debe ser un entero positivo.")
    if metadata.get("n_model_observations") is not None and n_markets and len(expected_months) >= 2:
        expected_obs = n_markets * (len(expected_months) - 2)
        if metadata.get("n_model_observations") != expected_obs:
            errors.append(
                "n_model_observations no coincide con la estructura del panel: "
                f"se esperaban {expected_obs} (= mercados × (meses - 2)); "
                f"recibido {metadata.get('n_model_observations')!r}."
            )

    series = payload.get("price_series")
    if not isinstance(series, list):
        return errors + ["price_series debe ser una lista."]
    expected_keys = {(month, product) for month in expected_months for product in products}
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    dates_by_product: dict[str, list[tuple[str, float]]] = {p: [] for p in products}

    for index, row in enumerate(series):
        if not isinstance(row, dict):
            errors.append(f"price_series[{index}] no es un objeto.")
            continue
        product = row.get("producto")
        raw_date = row.get("fecha")
        if not isinstance(product, str) or not product:
            errors.append(f"price_series[{index}] no tiene producto válido.")
            continue
        if not isinstance(raw_date, str):
            errors.append(f"price_series[{index}] no tiene fecha válida.")
            continue
        try:
            parsed_date = date.fromisoformat(raw_date[:10])
            month = f"{parsed_date.year:04d}-{parsed_date.month:02d}"
        except ValueError:
            errors.append(f"price_series[{index}] tiene fecha no interpretable: {raw_date!r}.")
            continue
        key = (month, product)
        if key in seen:
            errors.append(f"Registro duplicado para producto={product!r}, mes={month}.")
            continue
        seen[key] = row
        if key not in expected_keys:
            errors.append(f"Registro fuera del período/productos esperados: {product!r}, {month}.")

        price = row.get("precio_promedio_mercados")
        if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(float(price)):
            errors.append(f"Precio no numérico/faltante para {product!r}, {month}: {price!r}.")
            continue
        if float(price) <= 0:
            errors.append(f"Precio no positivo para {product!r}, {month}: {price!r}.")
            continue
        if product in dates_by_product:
            dates_by_product[product].append((month, float(price)))

        market_count = row.get("n_mercados")
        if market_count != n_markets:
            errors.append(
                f"Cobertura desigual para {product!r}, {month}: n_mercados={market_count!r}, "
                f"metadata.n_markets={n_markets}."
            )

    missing = expected_keys - set(seen)
    for month, product in sorted(missing):
        errors.append(f"Falta observación mensual para producto={product!r}, mes={month}.")

    if len(series) != len(expected_keys):
        errors.append(f"price_series contiene {len(series)} filas; se esperaban {len(expected_keys)}.")

    # Protege contra errores de escala como 2.500 -> 2.500 (en vez de 2500),
    # que producen saltos cercanos a x1000. Los saltos >10x requieren revisión y bloquean publicación.
    for product, points in dates_by_product.items():
        points.sort(key=lambda item: item[0])
        for (prev_month, prev_price), (month, price) in zip(points, points[1:]):
            ratio = price / prev_price
            if ratio < min_monthly_ratio or ratio > max_monthly_ratio:
                errors.append(
                    f"Posible cambio anómalo de escala en {product!r}: {prev_month}={prev_price:g}, "
                    f"{month}={price:g}, razón mensual={ratio:.4g}; permitido entre "
                    f"{min_monthly_ratio:g} y {max_monthly_ratio:g}. Revisar fuente antes de publicar."
                )

    markets = payload.get("markets")
    if isinstance(markets, list):
        names = [m.get("mercado") for m in markets if isinstance(m, dict)]
        if len(markets) != n_markets:
            errors.append(f"markets contiene {len(markets)} filas; metadata declara {n_markets} mercados.")
        if len(set(names)) != len(names):
            errors.append("markets contiene nombres repetidos.")
    else:
        errors.append("markets debe ser una lista.")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validar integridad del JSON de Agroclima.")
    parser.add_argument("json_path", nargs="?", default="data/observatorio.json", help="Ruta al JSON generado.")
    parser.add_argument("--start", default=os.getenv("AGROCLIMA_START_MONTH", f"{os.getenv('AGROCLIMA_START_YEAR', '2019')}-01"))
    parser.add_argument("--end", default=os.getenv("AGROCLIMA_END_MONTH", f"{os.getenv('AGROCLIMA_END_YEAR', '2024')}-12"))
    parser.add_argument("--min-monthly-ratio", type=float, default=float(os.getenv("AGROCLIMA_INTEGRITY_MIN_MONTHLY_RATIO", "0.1")))
    parser.add_argument("--max-monthly-ratio", type=float, default=float(os.getenv("AGROCLIMA_INTEGRITY_MAX_MONTHLY_RATIO", "10")))
    args = parser.parse_args()

    path = Path(args.json_path)
    if not path.is_file():
        print(f"ERROR integridad: no existe el archivo {path}.", file=sys.stderr)
        return 1
    try:
        payload = load_payload(path)
        errors = collect_integrity_errors(payload, args.start, args.end, args.min_monthly_ratio, args.max_monthly_ratio)
    except Exception as exc:
        print(f"ERROR integridad: no se pudo leer/validar JSON: {exc}", file=sys.stderr)
        return 1

    if errors:
        print(f"ERROR integridad: se encontraron {len(errors)} problema(s); se cancela la publicación:")
        for item in errors:
            print(f" - {item}")
        return 1

    metadata = payload["metadata"]
    print("OK integridad: JSON válido; publicación autorizada.")
    print(f" - Período: {metadata['start_date']} a {metadata['end_date']} ({metadata['n_months']} meses)")
    print(f" - Productos: {len(metadata['products'])}; mercados: {metadata['n_markets']}")
    print(f" - Observaciones de precios verificadas: {len(payload['price_series'])}")
    print(f" - Observaciones del modelo: {metadata.get('n_model_observations')}")
    print(f" - Saltos mensuales revisados: razón entre {args.min_monthly_ratio:g} y {args.max_monthly_ratio:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
