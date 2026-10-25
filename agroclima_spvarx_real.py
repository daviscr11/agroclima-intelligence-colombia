# -*- coding: utf-8 -*-
"""Herramientas para un SpVARX espacial de panel con datos reales SIPSA + NASA POWER.

No genera datos sintéticos. La unidad espacial es el mercado/ciudad observado en SIPSA,
no el departamento (SIPSA-P publica precios por mercado; véase el catálogo DANE).

Flujo recomendado desde Google Colab:
  1. Descargar archivos públicos SIPSA-P del DANE (microdatos 2013-2024) y ubicarlos en data/raw.
  2. load_sipsa_data(); revisar productos, nombres de columnas y auditoría.
  3. prepare_prices(); escoger productos con unidades/presentaciones comparables y panel balanceado.
  4. geocode_markets(); verificar manualmente market_coordinates.csv antes de estimar.
  5. fetch_nasa_power(); build_knn_weights(); build_model_panel(); estimate_spvarx(); compute_irfs().

La implementación estima un panel VARX espacial reducido con rezagos temporales y espaciales
rezagados: y_it = A y_i,t-1 + B W y_t-1 + Gamma x_it + Delta x_i,t-1 + efectos fijos.
No incorpora W*y_t contemporáneo (que exigiría un estimador simultáneo espacial distinto).
"""
from __future__ import annotations

import re
import time
import unicodedata
import zipfile
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt


# ----------------------- Normalización y lectura SIPSA -----------------------
MONTHS_ES = {
    "enero": 1, "ene": 1, "febrero": 2, "feb": 2, "marzo": 3, "mar": 3,
    "abril": 4, "abr": 4, "mayo": 5, "may": 5, "junio": 6, "jun": 6,
    "julio": 7, "jul": 7, "agosto": 8, "ago": 8, "septiembre": 9,
    "setiembre": 9, "sep": 9, "set": 9, "octubre": 10, "oct": 10,
    "noviembre": 11, "nov": 11, "diciembre": 12, "dic": 12,
}

COLUMN_CANDIDATES = {
    "market": ["mercado", "nombre mercado", "mercado mayorista", "ciudad", "municipio", "nombre municipio", "nom mpio", "fuente", "nombre fuente", "nombre de la fuente"],
    "product": ["producto", "nombre producto", "producto nombre", "nom producto", "alimento"],
    "price": ["precio promedio", "precio promedio mensual", "promedio precio", "precio medio", "precio", "valor precio"],
    "date": ["fecha", "fecha mes", "periodo", "periodo referencia", "fecha referencia"],
    "year": ["ano", "anio", "year", "vigencia"],
    "month": ["mes", "month", "mes referencia"],
    "unit": ["unidad medida", "unidad de medida", "unidad", "presentacion", "presentación", "unidad comercial"],
    "variety": ["variedad", "calidad", "variedad calidad", "tipo producto"],
}


def norm_text(value: Any) -> str:
    """Texto normalizado para comparar nombres sin acentos ni puntuación."""
    if pd.isna(value):
        return ""
    s = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9]+", " ", s).strip().lower()
    return re.sub(r"\s+", " ", s)


def _resolve_col(columns: Sequence[Any], role: str, column_map: Optional[Dict[str, Optional[str]]] = None) -> Optional[str]:
    column_map = column_map or {}
    explicit = column_map.get(role)
    if explicit:
        by_norm = {norm_text(c): c for c in columns}
        if norm_text(explicit) in by_norm:
            return by_norm[norm_text(explicit)]
        raise ValueError(f"COLUMN_MAP['{role}']={explicit!r} no existe. Columnas: {list(columns)}")
    normalized = {norm_text(c): c for c in columns}
    candidates = [norm_text(c) for c in COLUMN_CANDIDATES.get(role, [])]
    # Exact matches first; then allow a candidate phrase within a longer header.
    for cand in candidates:
        if cand in normalized:
            return normalized[cand]
    for cand in candidates:
        if cand:
            for ncol, original in normalized.items():
                if cand in ncol and len(cand) >= 4:
                    if role == "price" and any(x in ncol for x in ["minimo", "maximo", "variacion", "porcentaje"]):
                        continue
                    return original
    return None


def _parse_number(series: pd.Series) -> pd.Series:
    """Parsea números y precios SIPSA con separadores de miles/decimales.

    En archivos oficiales SIPSA-P hay precios en texto como ``2.947`` que significan
    2.947 pesos (dos mil novecientos cuarenta y siete), no 2,947 pesos decimales.
    Los puntos/comas que agrupan exactamente tres dígitos se interpretan como separadores
    de miles. Se conservan decimales genuinos como ``2.5`` y formatos mixtos como
    ``1.234,56`` o ``1,234.56``.
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    s = (series.astype(str).str.strip()
         .str.replace("$", "", regex=False)
         .str.replace(" ", "", regex=False))
    missing = s.str.lower().isin({"", "nan", "nat", "none", "null", "na", "<na>"})
    s.loc[missing] = np.nan

    both = s.str.contains(",", na=False) & s.str.contains(r"\.", na=False)
    comma_decimal = both & (s.str.rfind(",") > s.str.rfind("."))
    dot_decimal = both & ~comma_decimal

    # Formatos mixtos: eliminar el separador de miles y convertir el decimal a punto.
    s.loc[comma_decimal] = (s.loc[comma_decimal]
                            .str.replace(".", "", regex=False)
                            .str.replace(",", ".", regex=False))
    s.loc[dot_decimal] = s.loc[dot_decimal].str.replace(",", "", regex=False)

    # Separador único con grupos de tres dígitos: es miles, no decimal.
    dot_grouping = s.str.fullmatch(r"[+-]?\d{1,3}(?:\.\d{3})+", na=False)
    comma_grouping = s.str.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+", na=False)
    s.loc[dot_grouping] = s.loc[dot_grouping].str.replace(".", "", regex=False)
    s.loc[comma_grouping] = s.loc[comma_grouping].str.replace(",", "", regex=False)

    # Coma única no reconocida como agrupación: decimal, p. ej. ``2,5``.
    comma_only = s.str.contains(",", na=False) & ~s.str.contains(r"\.", na=False)
    s.loc[comma_only] = s.loc[comma_only].str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def _parse_date_column(series: pd.Series) -> pd.Series:
    """Parse dates in historic SIPSA tables, including Spanish labels such as ``ene-24``.

    DANE's recent annual files store the reference month as a Spanish abbreviation plus a
    two-digit year (e.g. ``ene-24``). Generic pandas/dateutil parsing does not reliably
    recognize that form, so parse month-year labels explicitly before using date fallbacks.
    Cached unique strings keep this inexpensive for large microdata files.
    """
    month_lookup = {norm_text(name): int(number) for name, number in MONTHS_ES.items()}

    def parse_one(value: Any) -> pd.Timestamp | pd.NaT:
        if value is None or pd.isna(value):
            return pd.NaT
        raw = str(value).strip()
        if not raw or raw.lower() in {"nan", "nat", "none", "null", "na"}:
            return pd.NaT
        ascii_value = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
        clean = re.sub(r"\s+", " ", ascii_value.strip().lower())

        # Month first: ene-24, septiembre 2024, set.24, etc.
        match = re.fullmatch(r"([a-z]+)[\s./_-]+(\d{2,4})", clean)
        if match:
            month_token, year_token = match.groups()
            month_num = month_lookup.get(norm_text(month_token))
            if month_num:
                year_num = int(year_token)
                if len(year_token) == 2:
                    year_num += 2000  # SIPSA-P catalogue years in this pipeline are 2013–2024.
                try:
                    return pd.Timestamp(year=year_num, month=month_num, day=1)
                except ValueError:
                    return pd.NaT

        # Numeric month-year forms: 01-24, 01/2024, 2024-01.
        match = re.fullmatch(r"(\d{1,2})[\s./_-]+(\d{2,4})", clean)
        if match:
            month_num, year_token = match.groups()
            month_num = int(month_num)
            year_num = int(year_token) + (2000 if len(year_token) == 2 else 0)
            if 1 <= month_num <= 12:
                try:
                    return pd.Timestamp(year=year_num, month=month_num, day=1)
                except ValueError:
                    return pd.NaT
        match = re.fullmatch(r"(\d{4})[\s./_-]+(\d{1,2})", clean)
        if match:
            year_num, month_num = int(match.group(1)), int(match.group(2))
            if 1 <= month_num <= 12:
                try:
                    return pd.Timestamp(year=year_num, month=month_num, day=1)
                except ValueError:
                    return pd.NaT

        # Explicit numeric formats first to avoid ambiguous inference.
        for fmt in (
            "%Y%m", "%Y%m%d", "%d%m%Y", "%m/%Y", "%m-%Y", "%m-%y",
            "%Y-%m", "%Y/%m", "%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y",
            "%d-%m-%y", "%Y-%m-%d", "%Y/%m/%d",
        ):
            parsed = pd.to_datetime(raw, format=fmt, errors="coerce", dayfirst=True)
            if pd.notna(parsed):
                return pd.Timestamp(parsed)

        # Fallback for Excel-like timestamps and other explicit ISO strings.
        try:
            parsed = pd.to_datetime(raw, errors="coerce", dayfirst=True)
            return pd.Timestamp(parsed) if pd.notna(parsed) else pd.NaT
        except (TypeError, ValueError, OverflowError):
            return pd.NaT

    raw_strings = series.astype(str)
    parsed_cache = {value: parse_one(value) for value in raw_strings.unique()}
    out = pd.to_datetime(raw_strings.map(parsed_cache), errors="coerce")
    return out.dt.to_period("M").dt.to_timestamp()


def _canonicalize_table(table: pd.DataFrame, source: str, column_map: Optional[Dict[str, Optional[str]]] = None) -> Optional[pd.DataFrame]:
    if table is None or table.empty:
        return None
    table = table.copy()
    table.columns = [str(c).strip() for c in table.columns]
    mapped = {role: _resolve_col(table.columns, role, column_map) for role in ["market", "product", "price", "date", "year", "month", "unit", "variety"]}
    if not all(mapped[r] for r in ["market", "product", "price"]):
        return None
    if not mapped["date"] and not (mapped["year"] and mapped["month"]):
        # Some year-by-year files expose only month; derive the year from filename if possible.
        year_match = re.search(r"20\d{2}", Path(source).name)
        if mapped["month"] and year_match:
            year_series = pd.Series(int(year_match.group(0)), index=table.index)
            month_series = table[mapped["month"]]
            mapped_year_from_file = year_series
        else:
            return None
    else:
        mapped_year_from_file = None

    if mapped["date"]:
        dates = _parse_date_column(table[mapped["date"]])
    else:
        year_source = table[mapped["year"]] if mapped["year"] else mapped_year_from_file
        year = pd.to_numeric(year_source, errors="coerce")
        month_raw = table[mapped["month"]].astype(str).str.strip().str.lower()
        month_text = month_raw.map(MONTHS_ES)
        month_num = pd.to_numeric(month_raw, errors="coerce").fillna(month_text)
        month_num = pd.to_numeric(month_num, errors="coerce")
        dates = pd.to_datetime({"year": year, "month": month_num, "day": 1}, errors="coerce")

    price = _parse_number(table[mapped["price"]])
    price_field_norm = norm_text(mapped["price"])
    # Recent DANE annual files express the dependent variable explicitly as average price
    # per kilogram and do not include a separate unit column. Treat the measurement unit as
    # kilogram for those rows; this is grounded in the published variable name, not imputed.
    price_is_per_kg = (
        "por kilogramo" in price_field_norm
        or "promedio kilogramo" in price_field_norm
        or "por kg" in price_field_norm
    )
    if price_is_per_kg:
        unit_values = pd.Series("Kilogramo", index=table.index, dtype="object")
    elif mapped["unit"]:
        unit_values = table[mapped["unit"]].astype(str).str.strip()
    else:
        unit_values = pd.Series("", index=table.index, dtype="object")

    out = pd.DataFrame({
        "fecha": dates,
        "mercado": table[mapped["market"]].astype(str).str.strip(),
        "producto": table[mapped["product"]].astype(str).str.strip(),
        "precio": price,
        "unidad": unit_values,
        "variedad": table[mapped["variety"]].astype(str).str.strip() if mapped["variety"] else "",
        "fuente_archivo": str(source),
    })
    out = out[out["fecha"].notna() & out["precio"].notna() & (out["precio"] > 0)].copy()
    out = out[out["mercado"].notna() & out["producto"].notna()]
    out = out[(out["mercado"].str.lower() != "nan") & (out["producto"].str.lower() != "nan")]
    return out if not out.empty else None


def _extract_archives(raw_dir: Path) -> None:
    """Extrae ZIPs descargados del DANE, incluidos ZIPs anidados, sin sobrescribir originales."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(4):
        archives = [p for p in raw_dir.rglob("*.zip") if "__extracted" not in p.parts]
        changed = False
        for archive in archives:
            target = archive.parent / f"{archive.stem}__extracted"
            marker = target / ".done"
            if marker.exists():
                continue
            target.mkdir(parents=True, exist_ok=True)
            try:
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(target)
                marker.write_text("ok", encoding="utf-8")
                changed = True
            except zipfile.BadZipFile:
                warnings.warn(f"Archivo no reconocido como ZIP: {archive}")
        if not changed:
            break


def _iter_tables(raw_dir: Path) -> Iterable[Tuple[str, pd.DataFrame]]:
    """Lee tablas SIPSA priorizando el CSV/TXT cuando existe su exportación equivalente.

    Algunas entregas anuales de DANE contienen el mismo registro en CSV y Stata (.dta).
    En casos como SIPSA-P 2024, el CSV conserva el token de precio como ``2.733`` (2.733
    pesos, con punto de miles), mientras que el .dta almacena ese mismo token como float
    ``2.733``. Si se corrige el CSV y se suma también el .dta, ambos registros pasan a
    parecer observaciones distintas y la media queda aproximadamente a la mitad.

    Por ello, si hay un CSV/TXT con el mismo nombre base normalizado en la misma carpeta,
    se considera la representación textual como fuente primaria y se omite el .dta duplicado.
    El .dta sigue siendo un respaldo cuando no hay archivo CSV/TXT equivalente.
    """
    exts = {".csv", ".txt", ".xlsx", ".xls", ".dta"}
    for path in raw_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        if path.name.startswith("~$") or path.name == ".done":
            continue
        if path.suffix.lower() == ".dta":
            # Exports like ``mensual 24.csv`` and ``Mensual_24.dta`` share a normalized
            # basename. Keep only the textual version so price tokens are parsed once.
            normalized_stem = norm_text(path.stem)
            text_sibling = any(
                sibling.is_file()
                and sibling.suffix.lower() in {".csv", ".txt"}
                and norm_text(sibling.stem) == normalized_stem
                for sibling in path.parent.iterdir()
            )
            if text_sibling:
                print(f"SIPSA: se omite exportación Stata duplicada {path.name}; se prioriza CSV/TXT equivalente.")
                continue
            try:
                yield str(path), pd.read_stata(path)
            except Exception as exc:
                warnings.warn(f"No se pudo leer Stata {path}: {exc}")
        elif path.suffix.lower() in {".xlsx", ".xls"}:
            try:
                sheets = pd.read_excel(path, sheet_name=None)
                for sheet_name, frame in sheets.items():
                    if isinstance(frame, pd.DataFrame) and not frame.empty:
                        yield f"{path}::{sheet_name}", frame
            except Exception as exc:
                warnings.warn(f"No se pudo leer Excel {path}: {exc}")
        else:
            frame = None
            for encoding in ["utf-8-sig", "latin-1", "cp1252"]:
                try:
                    frame = pd.read_csv(path, sep=None, engine="python", encoding=encoding, dtype=str)
                    if frame.shape[1] > 1:
                        break
                except Exception:
                    frame = None
            if frame is not None and not frame.empty:
                yield str(path), frame


def load_sipsa_data(
    raw_dir: str | Path,
    column_map: Optional[Dict[str, Optional[str]]] = None,
    table_audit_callback: Optional[Callable[[str, pd.DataFrame, Optional[pd.DataFrame]], None]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Lee archivos SIPSA-P y devuelve filas canónicas + auditoría resumida.

    No realiza imputación ni crea observaciones. COLUMN_MAP permite fijar los nombres exactos
    de las columnas cuando la estructura del ZIP cambia. `table_audit_callback`, si se aporta,
    recibe una vez cada tabla ya leída junto con sus filas canónicas (o None si no se reconoce),
    evitando volver a leer los archivos para generar auditorías detalladas.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"No existe {raw_dir}. Descargue archivos SIPSA-P de DANE y ubíquelos allí.")
    _extract_archives(raw_dir)
    frames, audit = [], []
    for source, table in _iter_tables(raw_dir):
        canonical = _canonicalize_table(table, source, column_map)
        if table_audit_callback is not None:
            table_audit_callback(source, table, canonical)
        if canonical is not None and not canonical.empty:
            frames.append(canonical)
            audit.append({"archivo_hoja": source, "filas_canonicas": len(canonical), "mercados": canonical["mercado"].nunique(), "productos": canonical["producto"].nunique(), "fecha_min": canonical["fecha"].min(), "fecha_max": canonical["fecha"].max()})
        else:
            # Informar una muestra de esquemas que requieren mapeo manual.
            if len(audit) < 8:
                audit.append({"archivo_hoja": source, "filas_canonicas": 0, "mercados": np.nan, "productos": np.nan, "fecha_min": np.nan, "fecha_max": np.nan, "columnas_detectadas": " | ".join(map(str, table.columns[:35]))})
    if not frames:
        audit_df = pd.DataFrame(audit)
        raise ValueError("No encontré una tabla SIPSA reconocible con mercado, producto, precio y fecha/mes-año. Revise los archivos extraídos y COLUMN_MAP.\n" + audit_df.to_string(index=False))
    data = pd.concat(frames, ignore_index=True)
    data["mercado_key"] = data["mercado"].map(norm_text)
    data["producto_key"] = data["producto"].map(norm_text)
    data["unidad_key"] = data["unidad"].map(norm_text)
    data["variedad_key"] = data["variedad"].map(norm_text)
    data = data.drop_duplicates(subset=["fecha", "mercado_key", "producto_key", "precio", "unidad_key", "variedad_key"]).reset_index(drop=True)
    audit_df = pd.DataFrame(audit)
    return data, audit_df


def summarize_sipsa(data: pd.DataFrame, top_n: int = 60) -> Dict[str, pd.DataFrame]:
    """Tablas para inspeccionar productos, unidades, variedades, mercados y cobertura."""
    products = (data.groupby("producto").agg(n=("precio", "size"), mercados=("mercado_key", "nunique"), meses=("fecha", "nunique"), fecha_min=("fecha", "min"), fecha_max=("fecha", "max")).sort_values(["mercados", "meses", "n"], ascending=False).head(top_n).reset_index())
    units = data.groupby(["producto", "unidad"], dropna=False).size().rename("n").reset_index().sort_values("n", ascending=False)
    varieties = data.groupby(["producto", "variedad"], dropna=False).size().rename("n").reset_index().sort_values("n", ascending=False)
    markets = data.groupby("mercado").agg(n=("precio", "size"), productos=("producto_key", "nunique"), meses=("fecha", "nunique"), fecha_min=("fecha", "min"), fecha_max=("fecha", "max")).sort_values("n", ascending=False).reset_index()
    return {"productos": products, "unidades": units, "variedades": varieties, "mercados": markets}


def prepare_prices(
    data: pd.DataFrame,
    products: Sequence[str],
    start_date: str = "2019-01-01",
    end_date: str = "2024-12-01",
    unit_per_product: Optional[Dict[str, str]] = None,
    variety_per_product: Optional[Dict[str, str]] = None,
    confirm_unitless: bool = False,
    min_markets: int = 6,
    min_months: int = 36,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], pd.DatetimeIndex, Dict[str, Any]]:
    """Filtra productos/unidades y selecciona el intervalo balanceado más largo que termina en end_date.

    No rellena meses ausentes. Si no hay panel balanceado suficiente, falla de forma explícita.
    """
    if len(products) < 2:
        raise ValueError("Un SpVARX necesita al menos dos variables endógenas. Configure PRODUCTS con 2 o más productos SIPSA.")
    if len({norm_text(p) for p in products}) != len(products):
        raise ValueError("PRODUCTS contiene nombres duplicados una vez normalizados.")
    unit_per_product = unit_per_product or {}
    variety_per_product = variety_per_product or {}
    product_lookup = {norm_text(p): p for p in data["producto"].dropna().unique()}
    chosen_norm = [norm_text(p) for p in products]
    missing = [p for p, n in zip(products, chosen_norm) if n not in product_lookup]
    if missing:
        options = data["producto"].value_counts().head(60).index.tolist()
        raise ValueError(f"No se encontraron estos nombres exactos de producto (sin acentos/puntuación): {missing}. Revise la tabla productos: {options}")
    actual_product = {norm_text(p): product_lookup[norm_text(p)] for p in products}
    # A partir de aquí, conservar las etiquetas exactas tal como aparecen en SIPSA.
    products = [actual_product[norm_text(p)] for p in products]
    chosen_norm = [norm_text(p) for p in products]
    selected = data[data["producto_key"].isin(chosen_norm)].copy()
    selected["producto_model"] = selected["producto_key"].map(actual_product)

    for product in products:
        pnorm = norm_text(product)
        subset = selected[selected["producto_key"] == pnorm]
        units = sorted(x for x in subset["unidad"].dropna().astype(str).unique() if x.strip() and norm_text(x) != "nan")
        if not subset["unidad"].map(norm_text).eq("").all() and len(units) > 1:
            chosen_unit = unit_per_product.get(product)
            if not chosen_unit:
                raise ValueError(f"El producto {product!r} tiene varias unidades/presentaciones: {units}. Configure UNIT_PER_PRODUCT[{product!r}] con la unidad comparable deseada.")
            selected = selected[~((selected["producto_key"] == pnorm) & (selected["unidad_key"] != norm_text(chosen_unit)))].copy()
        elif len(units) == 1:
            # Filtra explícitamente la única unidad no vacía si hay registros con unidad nula.
            selected = selected[~((selected["producto_key"] == pnorm) & (selected["unidad_key"] != norm_text(units[0])))].copy()
        elif not confirm_unitless:
            raise ValueError(f"No se detectó unidad comparable para {product!r}. Revise COLUMN_MAP['unit']; si el archivo ya contiene precios comparables, documente la validación y establezca CONFIRM_UNITLESS=True.")

        subset = selected[selected["producto_key"] == pnorm]
        varieties = sorted(x for x in subset["variedad"].dropna().astype(str).unique() if x.strip() and norm_text(x) != "nan")
        if len(varieties) > 1:
            chosen_variety = variety_per_product.get(product)
            if not chosen_variety:
                raise ValueError(f"El producto {product!r} tiene varias variedades/calidades: {varieties}. Configure VARIETY_PER_PRODUCT[{product!r}] para conservar una serie comparable.")
            selected = selected[~((selected["producto_key"] == pnorm) & (selected["variedad_key"] != norm_text(chosen_variety)))].copy()
        elif len(varieties) == 1:
            selected = selected[~((selected["producto_key"] == pnorm) & (selected["variedad_key"] != norm_text(varieties[0])))].copy()

    start, end = pd.Timestamp(start_date).to_period("M").to_timestamp(), pd.Timestamp(end_date).to_period("M").to_timestamp()
    selected = selected[selected["fecha"].between(start, end)].copy()
    if selected.empty:
        raise ValueError(f"No quedan observaciones entre {start:%Y-%m} y {end:%Y-%m}. Revise las fechas y la cobertura del archivo DANE.")
    selected["fecha"] = pd.to_datetime(selected["fecha"]).dt.to_period("M").dt.to_timestamp()
    monthly = (selected.groupby(["mercado", "mercado_key", "fecha", "producto_model"], as_index=False)
               .agg(precio=("precio", "mean"), n_observaciones=("precio", "size")))
    monthly = monthly.rename(columns={"producto_model": "producto"})
    monthly = monthly[monthly["precio"] > 0].copy()
    date_grid = pd.date_range(start, end, freq="MS")
    matrix = monthly.pivot_table(index=["mercado_key", "fecha"], columns="producto", values="precio", aggfunc="mean")
    present_markets = sorted(monthly["mercado_key"].unique())
    display_name = monthly.groupby("mercado_key")["mercado"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]).to_dict()

    selected_dates = None
    selected_markets: List[str] = []
    # Encuentra el sufijo más largo que termina en end_date con MIN_MONTHS y MIN_MARKETS completamente observados.
    for offset in range(max(1, len(date_grid) - min_months + 1)):
        candidate_dates = date_grid[offset:]
        if len(candidate_dates) < min_months:
            continue
        idx = pd.MultiIndex.from_product([present_markets, candidate_dates], names=["mercado_key", "fecha"])
        candidate = matrix.reindex(idx).reindex(columns=products)
        completeness = candidate.notna().all(axis=1).groupby(level="mercado_key").all()
        good = completeness[completeness].index.tolist()
        if len(good) >= min_markets:
            selected_dates, selected_markets = candidate_dates, sorted(good)
            break
    if selected_dates is None:
        # Produce a coverage diagnostic instead of silently filling missing values.
        coverage = (monthly.groupby(["mercado", "producto"])["fecha"].nunique().unstack("producto", fill_value=0))
        raise ValueError(f"No existe un panel balanceado con al menos {min_markets} mercados y {min_months} meses en el rango configurado. No se imputaron datos. Cobertura por mercado/producto:\n{coverage.to_string()}")

    idx = pd.MultiIndex.from_product([selected_markets, selected_dates], names=["mercado_key", "fecha"])
    balanced = matrix.reindex(idx).reindex(columns=products)
    balanced_long = balanced.stack(future_stack=True).rename("precio").reset_index().rename(columns={"producto": "producto"})
    balanced_long["mercado"] = balanced_long["mercado_key"].map(display_name)
    balanced_long = balanced_long[["mercado", "mercado_key", "fecha", "producto", "precio"]]
    display_markets = [display_name[k] for k in selected_markets]
    summary = {
        "input_observations": int(len(data)), "selected_observations": int(len(selected)),
        "monthly_market_product_rows": int(len(monthly)), "n_markets": int(len(selected_markets)),
        "n_months": int(len(selected_dates)), "start": selected_dates.min(), "end": selected_dates.max(),
        "products": list(products), "markets_display": display_markets, "markets_key": list(selected_markets),
        "unit_choices": unit_per_product, "variety_choices": variety_per_product,
        "nominal_prices": True,
    }
    return monthly, balanced_long, display_markets, selected_dates, summary


# ---------------------------- Coordenadas reales -----------------------------
def prepare_market_coordinates(markets: Sequence[str], coords_path: str | Path, auto_geocode: bool = True) -> pd.DataFrame:
    """Lee coordenadas validadas o geocodifica candidatos mediante Nominatim/OSM.

    La geocodificación inicial es solo una propuesta: inspeccione mercado_coordinates.csv
    y confirme el punto de cada mercado antes de realizar la estimación.
    """
    coords_path = Path(coords_path)
    coords_path.parent.mkdir(parents=True, exist_ok=True)
    required = {"mercado", "latitud", "longitud"}
    if coords_path.exists():
        coords = pd.read_csv(coords_path)
        if not required.issubset(coords.columns):
            raise ValueError(f"{coords_path} debe contener columnas {sorted(required)}")
    else:
        if not auto_geocode:
            template = pd.DataFrame({"mercado": list(markets), "latitud": np.nan, "longitud": np.nan, "nombre_geocodificado": ""})
            template.to_csv(coords_path, index=False, encoding="utf-8-sig")
            return template
        try:
            from geopy.geocoders import Nominatim
            from geopy.extra.rate_limiter import RateLimiter
        except ImportError as exc:
            raise ImportError("Instale geopy o cree manualmente market_coordinates.csv con mercado, latitud y longitud.") from exc
        geolocator = Nominatim(user_agent="agroclima-spvarx-colombia-research/1.0", timeout=25)
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.1, max_retries=2, error_wait_seconds=3.0, swallow_exceptions=True)
        rows = []
        for market in markets:
            location = geocode(f"{market}, Colombia", country_codes="co", exactly_one=True)
            rows.append({"mercado": market, "latitud": float(location.latitude) if location else np.nan,
                         "longitud": float(location.longitude) if location else np.nan,
                         "nombre_geocodificado": str(location.address) if location else "NO ENCONTRADO"})
        coords = pd.DataFrame(rows)
        coords.to_csv(coords_path, index=False, encoding="utf-8-sig")
        print(f"Se guardaron candidatos de geocodificación en {coords_path}. Revise y corrija el archivo antes de estimar.")
    coords["mercado_key"] = coords["mercado"].map(norm_text)
    coords = coords.drop_duplicates("mercado_key", keep="last")
    required_keys = {norm_text(x) for x in markets}
    missing = required_keys - set(coords["mercado_key"])
    if missing:
        raise ValueError(f"Faltan mercados en {coords_path}: {sorted(missing)}")
    subset = coords[coords["mercado_key"].isin(required_keys)].copy()
    if subset[["latitud", "longitud"]].isna().any().any():
        warnings.warn(f"Hay coordenadas faltantes en {coords_path}. La tabla se devuelve para que pueda verla y completarla manualmente.")
    if not subset["latitud"].between(-5.0, 14.0).all() or not subset["longitud"].between(-83.0, -65.0).all():
        warnings.warn("Hay coordenadas fuera del rango geográfico aproximado de Colombia; verifique el geocodificador/CSV.")
    return subset


# ----------------------------- NASA POWER real -------------------------------
def _power_point(market: str, latitude: float, longitude: float, start_year: int, end_year: int, timeout: int = 45) -> pd.DataFrame:
    url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
    params = {
        "parameters": "T2M,PRECTOTCORR", "community": "AG", "longitude": float(longitude),
        "latitude": float(latitude), "start": int(start_year), "end": int(end_year), "format": "JSON",
    }
    last_error = None
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            parameters = payload.get("properties", {}).get("parameter", {})
            if "T2M" not in parameters or "PRECTOTCORR" not in parameters:
                raise ValueError(f"NASA POWER no devolvió los parámetros esperados: {list(parameters)}; respuesta: {payload.get('messages', {})}")
            frames = []
            for variable in ["T2M", "PRECTOTCORR"]:
                block = pd.DataFrame({"periodo": list(parameters[variable].keys()), variable: list(parameters[variable].values())})
                block["fecha"] = pd.to_datetime(block["periodo"].astype(str), format="%Y%m", errors="coerce")
                frames.append(block[["fecha", variable]].set_index("fecha"))
            out = frames[0].join(frames[1], how="outer").reset_index()
            out["mercado"] = market
            out["latitud"] = latitude
            out["longitud"] = longitude
            out["T2M"] = pd.to_numeric(out["T2M"], errors="coerce")
            out["PRECTOTCORR"] = pd.to_numeric(out["PRECTOTCORR"], errors="coerce")
            out.loc[out["T2M"] <= -900, "T2M"] = np.nan
            out.loc[out["PRECTOTCORR"] <= -900, "PRECTOTCORR"] = np.nan
            return out
        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Falló consulta NASA POWER para {market!r}: {last_error}")


def fetch_nasa_power(coords: pd.DataFrame, start_date: str, end_date: str, cache_path: str | Path) -> pd.DataFrame:
    """Descarga clima mensual observado/modelado por NASA POWER; cachea por coordenadas exactas."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(start_date).to_period("M").to_timestamp()
    end = pd.Timestamp(end_date).to_period("M").to_timestamp()
    markets = coords["mercado"].tolist()
    needed_dates = pd.date_range(start, end, freq="MS")
    cache = pd.read_csv(cache_path, parse_dates=["fecha"]) if cache_path.exists() else pd.DataFrame()
    if not cache.empty:
        cache["mercado_key"] = cache["mercado"].map(norm_text)
    pulls = []
    for _, row in coords.iterrows():
        market = row["mercado"]
        key = norm_text(market)
        valid_cache = pd.DataFrame()
        if not cache.empty and key in set(cache["mercado_key"]):
            candidate = cache[cache["mercado_key"] == key].copy()
            candidate = candidate[(candidate["latitud"].sub(float(row["latitud"])).abs() < 1e-6) & (candidate["longitud"].sub(float(row["longitud"])).abs() < 1e-6)]
            valid_cache = candidate[candidate["fecha"].isin(needed_dates)]
        cached_dates = set(pd.to_datetime(valid_cache["fecha"]).tolist()) if not valid_cache.empty else set()
        if len(cached_dates) == len(needed_dates) and valid_cache[["T2M", "PRECTOTCORR"]].notna().all().all():
            pulls.append(valid_cache[["fecha", "mercado", "latitud", "longitud", "T2M", "PRECTOTCORR"]])
            print(f"NASA POWER: reutilizando caché de {market}")
        else:
            print(f"NASA POWER: descargando clima mensual para {market}")
            fresh = _power_point(market, row["latitud"], row["longitud"], start.year, end.year)
            pulls.append(fresh[fresh["fecha"].isin(needed_dates)])
    climate = pd.concat(pulls, ignore_index=True)
    climate = climate.drop_duplicates(["mercado", "fecha"], keep="last")
    climate.to_csv(cache_path, index=False, encoding="utf-8-sig")
    if climate[["T2M", "PRECTOTCORR"]].isna().any().any():
        bad = climate[climate[["T2M", "PRECTOTCORR"]].isna().any(axis=1)][["mercado", "fecha", "T2M", "PRECTOTCORR"]]
        raise ValueError(f"NASA POWER contiene valores faltantes; revise la fuente antes de estimar:\n{bad.head(20).to_string(index=False)}")
    climate["mercado_key"] = climate["mercado"].map(norm_text)
    return climate


# --------------------------- Matriz de pesos W -------------------------------
def build_knn_weights(coords: pd.DataFrame, k: int = 4):
    """Matriz KNN simétrica basada en distancia haversine real; estandarización por filas."""
    from sklearn.neighbors import NearestNeighbors
    from libpysal.weights import W
    coords = coords.copy()
    coords["mercado_key"] = coords["mercado"].map(norm_text)
    coords = coords.sort_values("mercado_key").reset_index(drop=True)
    markets = coords["mercado_key"].tolist()
    if len(markets) < 3:
        raise ValueError("Se necesitan al menos 3 mercados georreferenciados para construir W.")
    if coords[["latitud", "longitud"]].isna().any().any():
        raise ValueError("Hay coordenadas faltantes. Complete market_coordinates.csv y vuelva a revisar antes de construir W.")
    if not np.isfinite(coords[["latitud", "longitud"]].to_numpy(dtype=float)).all():
        raise ValueError("Hay coordenadas no finitas en la tabla de mercados.")
    k_eff = min(max(1, int(k)), len(markets) - 1)
    xy_rad = np.deg2rad(coords[["latitud", "longitud"]].to_numpy(dtype=float))
    model = NearestNeighbors(n_neighbors=k_eff + 1, metric="haversine", algorithm="ball_tree").fit(xy_rad)
    indices = model.kneighbors(xy_rad, return_distance=False)
    neighbor_sets = {markets[i]: set() for i in range(len(markets))}
    for i, neighbors in enumerate(indices):
        count = 0
        for j in neighbors:
            if int(j) == i:
                continue
            neighbor_sets[markets[i]].add(markets[int(j)])
            count += 1
            if count >= k_eff:
                break
    # Simetrizar para que la conexión sea mutua.
    for i, neighbors in list(neighbor_sets.items()):
        for j in list(neighbors):
            neighbor_sets[j].add(i)
    neighbors = {m: sorted(v) for m, v in neighbor_sets.items()}
    weights = {m: [1.0] * len(neighbors[m]) for m in markets}
    w = W(neighbors, weights, id_order=markets, silence_warnings=True)
    w.transform = "r"
    W_dense = np.asarray(w.full()[0], dtype=float)
    spectral_radius_w = float(np.max(np.abs(np.linalg.eigvals(W_dense))))
    return w, W_dense, markets, coords, {"method": "KNN haversine simétrico", "k": k_eff, "n": len(markets), "spectral_radius_W": spectral_radius_w}


# ---------------------- Panel y estimación SpVARX(1) -------------------------
def build_spvarx_panel(
    balanced_prices: pd.DataFrame,
    climate: pd.DataFrame,
    markets: Sequence[str],
    dates: pd.DatetimeIndex,
    products: Sequence[str],
    W_dense: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Construye y_it = Δlog(precio) y regresores locales/espaciales rezagados + clima.

    El modelo es un SpVARX(1) de panel en retornos logarítmicos; no rellena valores faltantes.
    """
    markets = list(markets)
    products = list(products)
    dates = pd.DatetimeIndex(dates)
    n, k = len(markets), len(products)
    if W_dense.shape != (n, n):
        raise ValueError("El orden/dimensión de W no coincide con la lista de mercados.")
    p = balanced_prices.copy()
    p["mercado_key"] = p["mercado_key"].map(norm_text)
    p["fecha"] = pd.to_datetime(p["fecha"]).dt.to_period("M").dt.to_timestamp()
    price_cube = []
    for dt in dates:
        frame = p[p["fecha"] == dt].pivot(index="mercado_key", columns="producto", values="precio").reindex(index=markets, columns=products)
        if frame.isna().any().any():
            raise ValueError(f"Panel de precios no balanceado en {dt:%Y-%m}; no se imputará automáticamente.")
        price_cube.append(frame.to_numpy(dtype=float))
    price_cube = np.stack(price_cube, axis=0)  # T x N x K
    if np.any(price_cube <= 0):
        raise ValueError("Se detectaron precios no positivos antes de logaritmos.")
    log_price = np.log(price_cube)
    returns = np.diff(log_price, axis=0)  # (T-1) x N x K; índice desde dates[1:]

    x = climate.copy()
    x["mercado_key"] = x["mercado_key"].map(norm_text)
    x["fecha"] = pd.to_datetime(x["fecha"]).dt.to_period("M").dt.to_timestamp()
    x = x[x["mercado_key"].isin(markets) & x["fecha"].isin(dates)].copy()
    x["calendar_month"] = x["fecha"].dt.month
    x["temp_anom_raw"] = x["T2M"] - x.groupby(["mercado_key", "calendar_month"])["T2M"].transform("mean")
    precip_log = np.log1p(x["PRECTOTCORR"].clip(lower=0))
    x["precip_anom_raw"] = precip_log - precip_log.groupby([x["mercado_key"], x["calendar_month"]]).transform("mean")
    for raw_name, model_name in [("temp_anom_raw", "temp_anom"), ("precip_anom_raw", "precip_anom")]:
        sd = float(x[raw_name].std(ddof=0))
        if not np.isfinite(sd) or sd == 0:
            raise ValueError(f"No hay variación suficiente en {raw_name} para estimarlo.")
        x[model_name] = (x[raw_name] - x[raw_name].mean()) / sd
    x_lookup = x.set_index(["fecha", "mercado_key"])[["temp_anom", "precip_anom"]]
    if x_lookup.index.has_duplicates:
        raise ValueError("Hay más de un dato NASA POWER por mercado y mes.")

    ycols = [f"y{i}" for i in range(k)]
    records = []
    # t empieza en 2 para construir Δlog(p)_t y su rezago Δlog(p)_{t-1}.
    for t in range(2, len(dates)):
        current_date, previous_date = dates[t], dates[t - 1]
        y_now = returns[t - 1]  # Δlog(P_t)
        y_lag = returns[t - 2]  # Δlog(P_{t-1})
        wy_lag = W_dense @ y_lag
        for i, market in enumerate(markets):
            x0 = x_lookup.loc[(current_date, market)]
            x1 = x_lookup.loc[(previous_date, market)]
            row = {"mercado_key": market, "fecha": current_date}
            for j in range(k):
                row[ycols[j]] = float(y_now[i, j])
                row[f"L1__y{j}"] = float(y_lag[i, j])
                row[f"WL1__y{j}"] = float(wy_lag[i, j])
            row["X0__temp_anom"] = float(x0["temp_anom"])
            row["X0__precip_anom"] = float(x0["precip_anom"])
            row["X1__temp_anom"] = float(x1["temp_anom"])
            row["X1__precip_anom"] = float(x1["precip_anom"])
            records.append(row)
    panel = pd.DataFrame(records).sort_values(["mercado_key", "fecha"]).reset_index(drop=True)
    if panel.isna().any().any():
        raise ValueError("El panel econométrico contiene faltantes tras alinear precio y clima. Revise fuentes y cobertura.")
    meta = {"markets": markets, "dates": dates, "products": products, "ycols": ycols, "n_markets": n, "n_products": k, "n_model_dates": panel["fecha"].nunique(), "n_obs": len(panel), "transform": "primera diferencia de log-precio", "climate_transform": "anomalía de mes calendario, estandarizada"}
    return panel, meta


def estimate_spvarx(panel: pd.DataFrame, products: Sequence[str], bandwidth: int = 4):
    """Estima un sistema SpVARX(1) de panel ecuación por ecuación con efectos fijos.

    Covarianza tipo Driscoll-Kraay (kernel Bartlett) para robustez frente a dependencia
    temporal y transversal. Los rezagos espaciales son W*y_{t-1}, no W*y_t simultáneo.
    """
    try:
        from linearmodels.panel import PanelOLS
    except ImportError as exc:
        raise ImportError("Instale linearmodels: pip install linearmodels") from exc
    products = list(products)
    ycols = [f"y{i}" for i in range(len(products))]
    regressors = [f"L1__y{i}" for i in range(len(products))] + [f"WL1__y{i}" for i in range(len(products))] + ["X0__temp_anom", "X0__precip_anom", "X1__temp_anom", "X1__precip_anom"]
    indexed = panel.set_index(["mercado_key", "fecha"]).sort_index()
    results, summaries, coefs = {}, [], []
    for j, product in enumerate(products):
        y = indexed[ycols[j]]
        X = indexed[regressors]
        model = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True, check_rank=True)
        fit = model.fit(cov_type="kernel", kernel="bartlett", bandwidth=int(bandwidth), debiased=True)
        results[product] = fit
        summaries.append(f"\n{'='*100}\nECUACIÓN: Δlog(precio) de {product}\n{'='*100}\n{fit.summary}\n")
        for name in fit.params.index:
            estimate = float(fit.params[name])
            se = float(fit.std_errors[name]) if name in fit.std_errors.index else np.nan
            pval = float(fit.pvalues[name]) if name in fit.pvalues.index else np.nan
            coefs.append({"producto_dependiente": product, "termino": name, "coeficiente": estimate, "error_estandar_DK": se, "p_value": pval, "ic95_inf": estimate - 1.96 * se if np.isfinite(se) else np.nan, "ic95_sup": estimate + 1.96 * se if np.isfinite(se) else np.nan, "n_obs": int(fit.nobs), "r2_within": float(fit.rsquared_within)})
    return results, pd.DataFrame(coefs), "\n".join(summaries)


def rolling_spvarx_backtest(
    balanced_prices: pd.DataFrame,
    climate: pd.DataFrame,
    markets: Sequence[str],
    dates: pd.DatetimeIndex,
    products: Sequence[str],
    W_dense: np.ndarray,
    min_train_months: int = 36,
) -> pd.DataFrame:
    """Backtest condicional de un paso, reestimando solo con meses pasados.

    La ecuación predictiva usa efectos fijos de mercado, pero omite efectos fijos de
    tiempo porque el efecto del mes objetivo no se conoce al emitir un pronóstico.
    La anomalía climática contemporánea se fija en cero (escenario de clima normal);
    la anomalía rezagada sí usa el último mes observado. No es una proyección de clima.
    """
    try:
        from linearmodels.panel import PanelOLS
    except ImportError as exc:
        raise ImportError("Instale linearmodels para ejecutar la validación temporal.") from exc

    market_order = [norm_text(m) for m in markets]
    product_order = list(products)
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).to_period("M").to_timestamp()
    if min_train_months < 24:
        raise ValueError("La validación temporal requiere al menos 24 meses iniciales de entrenamiento.")
    empty_columns = [
        "producto", "mercado_key", "fecha", "latitud", "longitud", "observado",
        "prediccion_spvarx", "prediccion_sin_cambio", "prediccion_persistencia",
        "meses_entrenamiento",
    ]
    if len(dates) <= min_train_months:
        return pd.DataFrame(columns=empty_columns)

    price_data = balanced_prices.copy()
    price_data["mercado_key"] = price_data["mercado_key"].map(norm_text)
    price_data["fecha"] = pd.to_datetime(price_data["fecha"]).dt.to_period("M").dt.to_timestamp()
    coordinates = climate[["mercado_key", "latitud", "longitud"]].copy()
    coordinates["mercado_key"] = coordinates["mercado_key"].map(norm_text)
    coordinates = coordinates.drop_duplicates("mercado_key").set_index("mercado_key").reindex(market_order)
    if coordinates[["latitud", "longitud"]].isna().any().any():
        raise ValueError("Faltan coordenadas de clima para uno o más mercados del backtest.")

    k = len(product_order)
    ycols = [f"y{i}" for i in range(k)]
    regressors = (
        [f"L1__y{i}" for i in range(k)]
        + [f"WL1__y{i}" for i in range(k)]
        + ["X0__temp_anom", "X0__precip_anom", "X1__temp_anom", "X1__precip_anom"]
    )

    def price_matrix(dt: pd.Timestamp) -> pd.DataFrame:
        block = price_data[price_data["fecha"] == dt]
        return block.pivot(index="mercado_key", columns="producto", values="precio").reindex(index=market_order, columns=product_order)

    predictions = []
    for test_idx in range(int(min_train_months), len(dates)):
        train_dates = dates[:test_idx]
        test_date = dates[test_idx]
        # Al pasar solo train_dates, medias estacionales y escalas climáticas se
        # estiman dentro del corte y no usan información del mes de validación.
        train_panel, _ = build_spvarx_panel(
            price_data, climate, market_order, train_dates, product_order, W_dense
        )
        origin_date = train_dates[-1]
        latest = train_panel[train_panel["fecha"] == origin_date].set_index("mercado_key").reindex(market_order)
        if len(latest) != len(market_order) or latest[ycols].isna().any().any():
            raise ValueError(f"No hay un vector rezagado completo para el origen {origin_date:%Y-%m}.")

        lag_values = latest[ycols].to_numpy(dtype=float)
        spatial_lag = W_dense @ lag_values
        test_X = pd.DataFrame(index=pd.Index(market_order, name="mercado_key"))
        for j in range(k):
            test_X[f"L1__y{j}"] = lag_values[:, j]
            test_X[f"WL1__y{j}"] = spatial_lag[:, j]
        # Clima contemporáneo desconocido al pronosticar: anomalía esperada cero.
        test_X["X0__temp_anom"] = 0.0
        test_X["X0__precip_anom"] = 0.0
        # El mes anterior ya se observó: su X0 en el panel pasa a ser X1 del test.
        test_X["X1__temp_anom"] = latest["X0__temp_anom"].to_numpy(dtype=float)
        test_X["X1__precip_anom"] = latest["X0__precip_anom"].to_numpy(dtype=float)

        previous_prices = price_matrix(origin_date)
        observed_prices = price_matrix(test_date)
        if previous_prices.isna().any().any() or observed_prices.isna().any().any():
            raise ValueError(f"Faltan precios completos para la validación de {test_date:%Y-%m}.")
        observed_returns = np.log(observed_prices.to_numpy(dtype=float) / previous_prices.to_numpy(dtype=float))

        train_indexed = train_panel.set_index(["mercado_key", "fecha"]).sort_index()
        for target_idx, product in enumerate(product_order):
            y = train_indexed[ycols[target_idx]]
            X = train_indexed[regressors]
            model = PanelOLS(
                y, X, entity_effects=True, time_effects=False,
                drop_absorbed=True, check_rank=True,
            )
            fit = model.fit(cov_type="unadjusted")
            beta = fit.params
            available_X = X.loc[:, beta.index]
            entity_effects = (y - available_X.dot(beta)).groupby(level=0).mean().reindex(market_order)
            prediction = test_X.loc[:, beta.index].dot(beta) + entity_effects.to_numpy(dtype=float)
            for market_idx, market in enumerate(market_order):
                predictions.append({
                    "producto": product,
                    "mercado_key": market,
                    "fecha": test_date,
                    "latitud": float(coordinates.loc[market, "latitud"]),
                    "longitud": float(coordinates.loc[market, "longitud"]),
                    "observado": float(observed_returns[market_idx, target_idx]),
                    "prediccion_spvarx": float(prediction.iloc[market_idx]),
                    "prediccion_sin_cambio": 0.0,
                    "prediccion_persistencia": float(lag_values[market_idx, target_idx]),
                    "meses_entrenamiento": int(len(train_dates)),
                })
    result = pd.DataFrame(predictions, columns=empty_columns)
    if not result.empty:
        result = result.sort_values(["producto", "fecha", "mercado_key"]).reset_index(drop=True)
    return result


def model_stability(results: Dict[str, Any], products: Sequence[str], W_dense: np.ndarray) -> Dict[str, Any]:
    """Radio espectral de la transición temporal-espacial del SpVARX(1)."""
    products = list(products)
    n, k = W_dense.shape[0], len(products)
    A = np.zeros((k, k), dtype=float)
    B = np.zeros((k, k), dtype=float)
    for r, product in enumerate(products):
        params = results[product].params
        for c in range(k):
            A[r, c] = float(params.get(f"L1__y{c}", 0.0))
            B[r, c] = float(params.get(f"WL1__y{c}", 0.0))
    transition = np.kron(A, np.eye(n)) + np.kron(B, W_dense)
    eigs = np.linalg.eigvals(transition)
    radius = float(np.max(np.abs(eigs)))
    return {"A_temporal": A, "B_espacial": B, "transition": transition, "spectral_radius": radius, "stable": bool(radius < 1.0), "eigenvalues": eigs}


def _irf_point_estimates(
    results: Dict[str, Any], products: Sequence[str], W_dense: np.ndarray, horizon: int
) -> pd.DataFrame:
    """Calcula las respuestas puntuales a un choque climático de una DE."""
    products = list(products)
    n, k = W_dense.shape[0], len(products)
    stability = model_stability(results, products, W_dense)
    M = stability["transition"]
    climate_vars = ["temp_anom", "precip_anom"]
    gamma0 = np.zeros((k, len(climate_vars)))
    gamma1 = np.zeros((k, len(climate_vars)))
    for j, product in enumerate(products):
        params = results[product].params
        for q, climate in enumerate(climate_vars):
            gamma0[j, q] = float(params.get(f"X0__{climate}", 0.0))
            gamma1[j, q] = float(params.get(f"X1__{climate}", 0.0))
    records = []
    for q, climate in enumerate(climate_vars):
        # Un impulso climático de un mes en un mercado a la vez; el promedio recorre todos los orígenes.
        origin_paths = []
        for source in range(n):
            first = np.zeros((n, k), dtype=float)
            first[source, :] = gamma0[:, q]
            path = [first]
            for step in range(1, int(horizon)):
                previous = path[-1]
                nxt = (M @ previous.reshape(-1, order="F")).reshape((n, k), order="F")
                if step == 1:
                    # El choque de x_t también entra vía x_{t-1} en y_{t+1}; después desaparece.
                    nxt[source, :] += gamma1[:, q]
                path.append(nxt)
            origin_paths.append(path)
        for h in range(int(horizon)):
            responses = np.stack([origin_paths[source][h] for source in range(n)], axis=0)  # origen x receptor x producto
            for out_j, out_product in enumerate(products):
                direct = float(np.mean([responses[s, s, out_j] for s in range(n)]))
                indirect_mean = float(np.mean([responses[s, np.arange(n) != s, out_j].mean() for s in range(n)])) if n > 1 else 0.0
                network_mean = float(np.mean(responses[:, :, out_j].mean(axis=1)))
                network_total = float(np.mean(responses[:, :, out_j].sum(axis=1)))
                records.append({"choque": climate, "horizonte": h, "producto_respuesta": out_product,
                                "directo": direct, "spillover_medio_receptores": indirect_mean,
                                "media_red": network_mean, "total_red_suma": network_total,
                                "radio_espectral_transicion": stability["spectral_radius"], "sistema_estable": stability["stable"]})
    return pd.DataFrame(records)


def _moving_block_indices(n_periods: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Índices de un bootstrap circular por bloques temporales."""
    blocks = []
    for _ in range(int(np.ceil(n_periods / block_length))):
        start = int(rng.integers(0, n_periods))
        blocks.append((start + np.arange(block_length)) % n_periods)
    return np.concatenate(blocks)[:n_periods]


def _bootstrap_spvarx_results(panel: pd.DataFrame, products: Sequence[str]) -> Dict[str, Any]:
    """Reestima el SpVARX de dos efectos fijos sobre un panel temporal remuestreado."""
    try:
        from linearmodels.panel import PanelOLS
    except ImportError as exc:
        raise ImportError("Instale linearmodels para calcular bandas bootstrap.") from exc

    products = list(products)
    ycols = [f"y{i}" for i in range(len(products))]
    regressors = (
        [f"L1__y{i}" for i in range(len(products))]
        + [f"WL1__y{i}" for i in range(len(products))]
        + ["X0__temp_anom", "X0__precip_anom", "X1__temp_anom", "X1__precip_anom"]
    )
    indexed = panel.set_index(["mercado_key", "fecha"]).sort_index()
    results = {}
    for j, product in enumerate(products):
        model = PanelOLS(
            indexed[ycols[j]], indexed[regressors], entity_effects=True,
            time_effects=True, drop_absorbed=True, check_rank=True,
        )
        results[product] = model.fit(cov_type="unadjusted")
    return results


def compute_irfs(
    results: Dict[str, Any],
    products: Sequence[str],
    W_dense: np.ndarray,
    horizon: int = 12,
    panel: Optional[pd.DataFrame] = None,
    bootstrap_repetitions: int = 0,
    bootstrap_seed: int = 20261025,
    block_length: Optional[int] = None,
) -> pd.DataFrame:
    """IRF puntuales e intervalos percentiles punto a punto del 95% por bootstrap.

    Cada bloque contiene todas las observaciones de los mercados en esos meses. Se
    conservan juntos y, dentro del bloque, en orden los resultados y regresores; en
    cada réplica se vuelve a estimar el SpVARX con efectos de mercado y tiempo.
    Las bandas son condicionales a W, la cobertura y los regresores observados, son
    aproximadas, no son simultáneas para toda la curva, y no resuelven el sesgo
    potencial del panel dinámico ni identifican una respuesta causal.
    """
    point = _irf_point_estimates(results, products, W_dense, horizon)
    if panel is None or int(bootstrap_repetitions) <= 0:
        return point

    repetitions = int(bootstrap_repetitions)
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(panel["fecha"]).unique()))
    if len(dates) < 12:
        raise ValueError("El bootstrap IRF requiere al menos 12 meses de estimación.")
    if block_length is None:
        block_length = max(2, int(round(len(dates) ** (1 / 3))))
    block_length = min(int(block_length), len(dates))
    if block_length < 2:
        raise ValueError("La longitud del bloque bootstrap debe ser de al menos 2 meses.")

    source_blocks = {
        pd.Timestamp(date): block.copy()
        for date, block in panel.assign(fecha=pd.to_datetime(panel["fecha"])).groupby("fecha", sort=True)
    }
    anchor = pd.Timestamp("2000-01-01")
    rng = np.random.default_rng(int(bootstrap_seed))
    keys = ["choque", "horizonte", "producto_respuesta"]
    metrics = ["directo", "spillover_medio_receptores", "media_red", "total_red_suma"]
    draws = []

    for _ in range(repetitions):
        sampled_indices = _moving_block_indices(len(dates), block_length, rng)
        sampled_blocks = []
        for new_period, source_index in enumerate(sampled_indices):
            source_date = pd.Timestamp(dates[int(source_index)])
            block = source_blocks[source_date].copy()
            block["fecha"] = anchor + pd.DateOffset(months=new_period)
            sampled_blocks.append(block)
        sampled_panel = pd.concat(sampled_blocks, ignore_index=True)
        try:
            draw_results = _bootstrap_spvarx_results(sampled_panel, products)
            draw = _irf_point_estimates(draw_results, products, W_dense, horizon)
            if np.isfinite(draw[metrics].to_numpy(dtype=float)).all():
                draws.append(draw)
        except Exception:
            # Una réplica singular o no estimable se descarta y queda contabilizada.
            continue

    valid = len(draws)
    minimum_valid = max(50, int(np.ceil(0.8 * repetitions)))
    point["n_bootstrap_valid"] = valid
    point["n_bootstrap_requested"] = repetitions
    point["bootstrap_block_months"] = block_length
    point["bootstrap_method"] = "Bootstrap circular móvil por bloques temporales; percentiles 2,5–97,5%"
    distribution = pd.concat(draws, ignore_index=True) if valid >= minimum_valid else None
    for metric in metrics:
        lower_name, upper_name = f"{metric}_ic95_inf", f"{metric}_ic95_sup"
        point[lower_name] = np.nan
        point[upper_name] = np.nan
        if distribution is None:
            continue
        quantiles = distribution.groupby(keys)[metric].quantile([0.025, 0.975]).unstack(-1)
        quantiles = quantiles.rename(columns={0.025: lower_name, 0.975: upper_name}).reset_index()
        point = point.drop(columns=[lower_name, upper_name]).merge(quantiles, on=keys, how="left")
    point["bootstrap_interval_available"] = valid >= minimum_valid
    return point


def _panel_with_spatial_lags(
    panel: pd.DataFrame, products: Sequence[str], markets: Sequence[str], W_dense: np.ndarray
) -> pd.DataFrame:
    """Recalcula W·y(t−1) sin alterar resultados, clima ni muestra temporal."""
    products = list(products)
    markets = list(markets)
    if W_dense.shape != (len(markets), len(markets)):
        raise ValueError("La matriz de pesos no coincide con el orden de mercados del panel.")
    out = panel.copy()
    y_lag_cols = [f"L1__y{j}" for j in range(len(products))]
    spatial_cols = [f"WL1__y{j}" for j in range(len(products))]
    for _, group in out.groupby("fecha", sort=True):
        ordered = group.set_index("mercado_key").reindex(markets)
        lag_values = ordered[y_lag_cols].to_numpy(dtype=float)
        if lag_values.shape[0] != len(markets) or not np.isfinite(lag_values).all():
            raise ValueError("Faltan rezagos de precios para reconstruir los rezagos espaciales.")
        spatial_values = W_dense @ lag_values
        for j, col in enumerate(spatial_cols):
            mapping = dict(zip(markets, spatial_values[:, j]))
            out.loc[group.index, col] = group["mercado_key"].map(mapping).to_numpy(dtype=float)
    return out


def spatial_weights_sensitivity(
    panel: pd.DataFrame,
    coords: pd.DataFrame,
    products: Sequence[str],
    k_values: Sequence[int] = (3, 4, 5),
    base_k: int = 4,
    base_weights: Optional[np.ndarray] = None,
    base_results: Optional[Dict[str, Any]] = None,
    base_coefficients: Optional[pd.DataFrame] = None,
    base_irfs: Optional[pd.DataFrame] = None,
    horizon: int = 12,
    bootstrap_repetitions: int = 0,
    bootstrap_seed: int = 20261025,
) -> Dict[str, pd.DataFrame]:
    """Compara la misma muestra SpVARX bajo distintas matrices KNN simétricas.

    Si se pasan los resultados de la especificación base, esa estimación se reutiliza.
    Cada alternativa reconstruye solo W·y(t−1), reestima los mismos productos y
    efectos fijos, y calcula IRF con la misma longitud y bootstrap que la base.
    """
    k_values = list(dict.fromkeys(int(k) for k in k_values))
    if not k_values or any(k < 1 for k in k_values):
        raise ValueError("k_values debe contener enteros positivos.")
    if base_k not in k_values:
        raise ValueError("La comparación debe incluir la matriz KNN base.")

    summary_rows, coefficient_frames, irf_frames = [], [], []
    for k in k_values:
        requested_k = int(k)
        try:
            _, candidate_W, markets, _, weight_meta = build_knn_weights(coords, k=requested_k)
            use_base = requested_k == int(base_k) and base_results is not None and base_coefficients is not None and base_irfs is not None
            if use_base:
                if base_weights is not None and not np.allclose(candidate_W, base_weights, atol=1e-12, rtol=0):
                    raise ValueError("La matriz reconstruida para k base no coincide con la matriz estimada originalmente.")
                if base_weights is not None:
                    candidate_W = np.asarray(base_weights, dtype=float)
                candidate_panel = panel
                candidate_results = base_results
                candidate_coefficients = base_coefficients
                candidate_irfs = base_irfs
            else:
                candidate_panel = _panel_with_spatial_lags(panel, products, markets, candidate_W)
                bandwidth = min(4, max(1, candidate_panel["fecha"].nunique() - 1))
                candidate_results, candidate_coefficients, _ = estimate_spvarx(candidate_panel, products, bandwidth=bandwidth)
                candidate_irfs = compute_irfs(
                    candidate_results, products, candidate_W, horizon=horizon,
                    panel=candidate_panel, bootstrap_repetitions=bootstrap_repetitions,
                    bootstrap_seed=bootstrap_seed,
                )

            stability = model_stability(candidate_results, products, candidate_W)
            spatial_coefs = candidate_coefficients[
                candidate_coefficients["termino"].astype(str).str.startswith("WL1__y")
            ].copy()
            spatial_coefs["k_vecinos"] = requested_k
            spatial_coefs["k_efectivo"] = int(weight_meta["k"])
            spatial_coefs["especificacion_base"] = requested_k == int(base_k)
            coefficient_frames.append(spatial_coefs)

            irf_data = candidate_irfs.copy()
            irf_data["k_vecinos"] = requested_k
            irf_data["k_efectivo"] = int(weight_meta["k"])
            irf_data["especificacion_base"] = requested_k == int(base_k)
            irf_frames.append(irf_data)
            valid_replicates = (
                int(irf_data["n_bootstrap_valid"].max())
                if "n_bootstrap_valid" in irf_data and not irf_data.empty else 0
            )
            intervals_available = (
                bool(irf_data["bootstrap_interval_available"].all())
                if "bootstrap_interval_available" in irf_data and not irf_data.empty else False
            )
            summary_rows.append({
                "k_vecinos": requested_k,
                "k_efectivo": int(weight_meta["k"]),
                "especificacion_base": requested_k == int(base_k),
                "n_mercados": int(weight_meta["n"]),
                "radio_espectral_W": float(weight_meta["spectral_radius_W"]),
                "radio_espectral_transicion": float(stability["spectral_radius"]),
                "sistema_estable": bool(stability["stable"]),
                "replicas_bootstrap_validas": valid_replicates,
                "intervalos_irf_disponibles": intervals_available,
                "estado": "Estimación disponible",
                "error": None,
            })
        except Exception as exc:
            summary_rows.append({
                "k_vecinos": requested_k, "k_efectivo": None,
                "especificacion_base": requested_k == int(base_k),
                "n_mercados": int(panel["mercado_key"].nunique()),
                "radio_espectral_W": None, "radio_espectral_transicion": None,
                "sistema_estable": None, "replicas_bootstrap_validas": 0,
                "intervalos_irf_disponibles": False,
                "estado": "No disponible", "error": str(exc),
            })

    return {
        "summary": pd.DataFrame(summary_rows),
        "coefficients": pd.concat(coefficient_frames, ignore_index=True) if coefficient_frames else pd.DataFrame(),
        "irfs": pd.concat(irf_frames, ignore_index=True) if irf_frames else pd.DataFrame(),
    }


# ------------------------------ Diagnósticos ---------------------------------
def stationarity_audit(panel: pd.DataFrame, products: Sequence[str]) -> pd.DataFrame:
    from statsmodels.tsa.stattools import adfuller
    rows = []
    for j, product in enumerate(products):
        for market, sub in panel.groupby("mercado_key"):
            values = sub.sort_values("fecha")[f"y{j}"].dropna().to_numpy()
            try:
                stat, pval, *_ = adfuller(values, maxlag=min(6, max(0, len(values)//5)), regression="c", autolag="AIC")
            except Exception:
                stat, pval = np.nan, np.nan
            rows.append({"producto": product, "mercado_key": market, "ADF_stat": stat, "ADF_pvalue": pval, "rechaza_raiz_unitaria_5pct": bool(pval < 0.05) if np.isfinite(pval) else False, "n": len(values)})
    return pd.DataFrame(rows)


def residual_moran_panel_audit(
    results: Dict[str, Any],
    W,
    markets: Sequence[str],
    permutations: int = 4999,
    seed: int = 20261024,
) -> pd.DataFrame:
    """Calcula Moran global de residuos para cada producto y cada fecha del panel.

    Incluye p-valores simulados sin ajuste y correcciones FDR-BH / FDR-BY por
    producto para las pruebas temporales. Si faltan residuos en alguno de los
    mercados de una fecha, esa fila se marca como incompleta en lugar de calcular
    Moran con una matriz W desalineada.
    """
    from esda.moran import Moran
    from statsmodels.stats.multitest import multipletests

    if permutations < 999:
        raise ValueError("Use al menos 999 permutaciones; se recomiendan 4999 para la auditoría.")

    w_order_raw = getattr(W, "id_order", None)
    ordered_keys = [norm_text(m) for m in (w_order_raw if w_order_raw else markets)]
    market_keys = [norm_text(m) for m in markets]

    if len(ordered_keys) != len(set(ordered_keys)):
        raise ValueError("La matriz espacial W contiene identificadores duplicados.")
    if set(ordered_keys) != set(market_keys):
        raise ValueError("Los mercados de W no coinciden con los mercados de los residuos.")
    if len(ordered_keys) < 3:
        raise ValueError("Se necesitan al menos 3 mercados para calcular Moran global.")

    frames: Dict[str, pd.DataFrame] = {}
    all_dates = set()
    for product, fit in results.items():
        resid = fit.resids
        temp = resid.rename("residuo").reset_index()
        if temp.shape[1] < 3:
            raise ValueError(f"No se pudieron identificar mercado y fecha en los residuos de {product!r}.")
        entity_col, time_col = temp.columns[0], temp.columns[1]
        temp = temp.rename(columns={entity_col: "mercado_key", time_col: "fecha"})
        temp["mercado_key"] = temp["mercado_key"].map(norm_text)
        temp["fecha"] = pd.to_datetime(temp["fecha"], errors="coerce")
        temp["residuo"] = pd.to_numeric(temp["residuo"], errors="coerce")
        if temp["fecha"].isna().any():
            raise ValueError(f"Hay fechas inválidas en los residuos de {product!r}.")
        if temp.duplicated(["mercado_key", "fecha"]).any():
            raise ValueError(f"Hay residuos duplicados por mercado y fecha en {product!r}.")
        frames[product] = temp
        all_dates.update(pd.Timestamp(d) for d in temp["fecha"].unique())

    if not all_dates:
        raise ValueError("No hay fechas de residuos disponibles para la auditoría de Moran.")

    records = []
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        for product, temp in frames.items():
            by_date = {
                pd.Timestamp(date): block.set_index("mercado_key")["residuo"]
                for date, block in temp.groupby("fecha")
            }
            for date in sorted(all_dates):
                values_series = by_date.get(date, pd.Series(dtype=float))
                values = values_series.reindex(ordered_keys).to_numpy(dtype=float)
                n_valid = int(np.isfinite(values).sum())
                row = {
                    "producto_dependiente": product,
                    "fecha_residuo": pd.Timestamp(date),
                    "Moran_I_residuos": np.nan,
                    "p_sim": np.nan,
                    "p_ajustado_BH": np.nan,
                    "p_ajustado_BY": np.nan,
                    "significativo_5pct_sin_ajuste": False,
                    "significativo_5pct_BH": False,
                    "significativo_5pct_BY": False,
                    "n": n_valid,
                    "n_esperado": len(ordered_keys),
                    "permutaciones": int(permutations),
                    "estado": "OK",
                }
                if n_valid != len(ordered_keys):
                    row["estado"] = "INCOMPLETO: faltan residuos para uno o más mercados"
                    records.append(row)
                    continue
                if np.allclose(values, values[0]):
                    row["estado"] = "NO CALCULABLE: residuos constantes"
                    records.append(row)
                    continue
                try:
                    mi = Moran(values, W, permutations=int(permutations), two_tailed=True)
                    row["Moran_I_residuos"] = float(mi.I)
                    row["p_sim"] = float(mi.p_sim)
                except Exception as exc:
                    row["estado"] = f"ERROR: {type(exc).__name__}: {exc}"
                records.append(row)
    finally:
        np.random.set_state(rng_state)

    audit = pd.DataFrame(records)
    for product in audit["producto_dependiente"].dropna().unique():
        mask = (
            (audit["producto_dependiente"] == product)
            & (audit["estado"] == "OK")
            & audit["p_sim"].notna()
        )
        indexes = audit.index[mask]
        if len(indexes) == 0:
            continue
        p_values = audit.loc[indexes, "p_sim"].to_numpy(dtype=float)
        reject_bh, adjusted_bh, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")
        reject_by, adjusted_by, _, _ = multipletests(p_values, alpha=0.05, method="fdr_by")
        audit.loc[indexes, "p_ajustado_BH"] = adjusted_bh
        audit.loc[indexes, "p_ajustado_BY"] = adjusted_by
        audit.loc[indexes, "significativo_5pct_sin_ajuste"] = p_values < 0.05
        audit.loc[indexes, "significativo_5pct_BH"] = reject_bh
        audit.loc[indexes, "significativo_5pct_BY"] = reject_by

    return audit.sort_values(["producto_dependiente", "fecha_residuo"]).reset_index(drop=True)


def residual_moran(results: Dict[str, Any], W, markets: Sequence[str]) -> pd.DataFrame:
    """Audita Moran en todas las fechas, guarda el CSV y devuelve Moran del último mes.

    La tabla devuelta conserva las columnas/filas esperadas por el JSON del observatorio;
    el detalle temporal completo se guarda aparte en data/residual_moran_panel_audit.csv.
    """
    permutations = 4999
    audit = residual_moran_panel_audit(
        results, W, markets, permutations=permutations, seed=20261024
    )

    audit_path = Path(__file__).resolve().parent / "data" / "residual_moran_panel_audit.csv"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    print(
        f"[Moran audit] Guardadas {len(audit)} filas en {audit_path}; "
        f"permutaciones por prueba={permutations}."
    )
    for product, group in audit.groupby("producto_dependiente", sort=False):
        valid = group[(group["estado"] == "OK") & group["p_sim"].notna()]
        raw_sig = int(valid["significativo_5pct_sin_ajuste"].sum())
        bh_sig = int(valid["significativo_5pct_BH"].sum())
        by_sig = int(valid["significativo_5pct_BY"].sum())
        print(
            f"[Moran audit] {product}: meses válidos={len(valid)}/{len(group)}; "
            f"significativos sin ajuste={raw_sig}; tras BH={bh_sig}; tras BY={by_sig}; "
            f"Moran medio={valid['Moran_I_residuos'].mean():.4f}"
        )

    valid_latest = audit[(audit["estado"] == "OK") & audit["p_sim"].notna()]
    if valid_latest.empty:
        raise RuntimeError("No se pudo calcular Moran para ninguna fecha; revise residual_moran_panel_audit.csv.")

    rows = []
    for product in results:
        product_rows = valid_latest[valid_latest["producto_dependiente"] == product]
        if product_rows.empty:
            rows.append({
                "producto_dependiente": product,
                "fecha_residuo": pd.NaT,
                "Moran_I_residuos": np.nan,
                "p_sim": np.nan,
                "n": 0,
            })
            continue
        latest = product_rows.loc[product_rows["fecha_residuo"].idxmax()]
        rows.append({
            "producto_dependiente": product,
            "fecha_residuo": latest["fecha_residuo"],
            "Moran_I_residuos": float(latest["Moran_I_residuos"]),
            "p_sim": float(latest["p_sim"]),
            "n": int(latest["n"]),
        })
    return pd.DataFrame(rows)


# --------------------------------- Gráficos ----------------------------------
def plot_nominal_prices(balanced_prices: pd.DataFrame, products: Sequence[str]):
    products = list(products)
    fig, axes = plt.subplots(len(products), 1, figsize=(12, max(3.8, 3.4 * len(products))), squeeze=False)
    for ax, product in zip(axes[:, 0], products):
        sub = balanced_prices[balanced_prices["producto"] == product]
        avg = sub.groupby("fecha")["precio"].mean()
        ax.plot(avg.index, avg.values, linewidth=1.8)
        ax.set_title(f"Precio SIPSA nominal promedio entre mercados — {product}")
        ax.set_ylabel("Precio en unidad reportada por SIPSA")
        ax.grid(True, alpha=0.25)
    axes[-1, 0].set_xlabel("Mes")
    fig.tight_layout()
    return fig


def plot_market_growth_map(panel: pd.DataFrame, coords: pd.DataFrame, product_index: int = 0):
    ycol = f"y{product_index}"
    latest_date = panel["fecha"].max()
    latest = panel[panel["fecha"] == latest_date][["mercado_key", ycol]]
    xy = coords.copy()
    xy["mercado_key"] = xy["mercado"].map(norm_text)
    merged = xy.merge(latest, on="mercado_key", how="inner")
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(merged["longitud"], merged["latitud"], c=merged[ycol], s=95, edgecolor="black", linewidth=0.4, cmap="RdYlBu_r")
    for _, row in merged.iterrows():
        ax.annotate(row["mercado"], (row["longitud"], row["latitud"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
    fig.colorbar(sc, ax=ax, label="Δ log(precio), aproximadamente variación mensual")
    ax.set_title(f"Cambio mensual del precio por mercado — {latest_date:%Y-%m}\nPuntos geográficos; no representa límites administrativos")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def plot_coefficients(coef_df: pd.DataFrame, dependent_product: str):
    sub = coef_df[coef_df["producto_dependiente"] == dependent_product].copy()
    if sub.empty:
        raise ValueError(f"No hay coeficientes para {dependent_product}")
    sub = sub.sort_values("coeficiente")
    fig, ax = plt.subplots(figsize=(10, max(4, 0.42 * len(sub))))
    y = np.arange(len(sub))
    xerr = np.vstack([sub["coeficiente"] - sub["ic95_inf"], sub["ic95_sup"] - sub["coeficiente"]])
    ax.errorbar(sub["coeficiente"], y, xerr=xerr, fmt="o", capsize=3)
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_yticks(y, sub["termino"])
    ax.set_title(f"Coeficientes e IC 95% (Driscoll–Kraay) — ecuación {dependent_product}")
    ax.set_xlabel("Coeficiente en Δ log(precio)")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_irfs(irf_df: pd.DataFrame, shock: str = "temp_anom"):
    sub = irf_df[irf_df["choque"] == shock]
    products = sub["producto_respuesta"].unique().tolist()
    fig, axes = plt.subplots(len(products), 1, figsize=(11, max(4, 3.7 * len(products))), squeeze=False)
    for ax, product in zip(axes[:, 0], products):
        s = sub[sub["producto_respuesta"] == product].sort_values("horizonte")
        ax.plot(s["horizonte"], s["directo"], marker="o", label="Directo: mercado afectado")
        ax.plot(s["horizonte"], s["spillover_medio_receptores"], marker="s", label="Spillover medio por receptor")
        ax.plot(s["horizonte"], s["media_red"], marker="^", label="Media de toda la red")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"Respuesta de {product} a choque +1 DE de {shock}")
        ax.set_xlabel("Horizonte mensual")
        ax.set_ylabel("Respuesta en Δ log(precio)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def save_outputs(output_dir: str | Path, **objects) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in objects.items():
        path = output_dir / name
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(path, index=False, encoding="utf-8-sig")
        elif isinstance(obj, str):
            path.write_text(obj, encoding="utf-8")
    return output_dir
