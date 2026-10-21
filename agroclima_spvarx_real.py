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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
    # En archivos SIPSA-P antiguos, la variable "Fuente" identifica el mercado
    # informado. Si también existe "Mercado", el match exacto de "Mercado" gana.
    "market": ["mercado", "nombre mercado", "mercado mayorista", "ciudad", "municipio", "nombre municipio", "nom mpio", "fuente", "fuente mercado"],
    "product": ["producto", "nombre producto", "producto nombre", "nom producto", "alimento"],
    "price": ["precio promedio por kilogramo", "precio por kilogramo", "precio promedio por kilo", "precio por kilo", "precio promedio mensual", "precio promedio", "promedio precio", "precio medio", "precio", "valor precio"],
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
    """Convierte números con formato decimal colombiano sin reinterpretar decimales con punto."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype(str).str.strip().str.replace("$", "", regex=False).str.replace(" ", "", regex=False)
    both = s.str.contains(",", na=False) & s.str.contains(r"\.", na=False)
    comma_only = s.str.contains(",", na=False) & ~s.str.contains(r"\.", na=False)
    # Si hay punto y coma, en la mayor parte de los formatos locales el último separador es decimal.
    comma_decimal = both & (s.str.rfind(",") > s.str.rfind("."))
    dot_decimal = both & ~comma_decimal
    s.loc[comma_decimal] = s.loc[comma_decimal].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    s.loc[dot_decimal] = s.loc[dot_decimal].str.replace(",", "", regex=False)
    s.loc[comma_only] = s.loc[comma_only].str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def _parse_date_column(series: pd.Series) -> pd.Series:
    """Parsea fechas mensuales heterogéneas del DANE sin asumir un único formato.

    Soporta fechas Excel/Pandas, YYYYMM, YYYYMMDD, YYYY-MM, YYYY-M01/2024M01,
    MM/YYYY, DD/MM/YYYY y meses en español. Devuelve el primer día del mes.
    """
    s = series.copy()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    # Las columnas que pandas ya leyó como fechas se preservan sin convertirlas a texto.
    try:
        if pd.api.types.is_datetime64_any_dtype(s):
            return pd.to_datetime(s, errors="coerce").dt.to_period("M").dt.to_timestamp()
    except Exception:
        pass

    # Reconocer números almacenados como fecha Excel o como códigos YYYYMM/YYYMMDD.
    numeric = pd.to_numeric(s, errors="coerce")
    numeric_mask = numeric.notna()
    if numeric_mask.any():
        n_as_int = numeric[numeric_mask].astype("Int64").astype(str)
        idx = n_as_int.index
        compact = n_as_int.str.strip()
        m6 = compact.str.fullmatch(r"(?:19|20)\d{4}")
        if m6.any():
            vals = compact[m6]
            out.loc[vals.index] = pd.to_datetime(vals, format="%Y%m", errors="coerce")
        m8 = compact.str.fullmatch(r"(?:19|20)\d{6}")
        if m8.any():
            vals = compact[m8]
            out.loc[vals.index] = pd.to_datetime(vals, format="%Y%m%d", errors="coerce")
        # Seriales Excel habituales. Excluir enteros identificados arriba.
        serial_mask = numeric_mask & out.isna() & numeric.between(30000, 70000)
        if serial_mask.any():
            out.loc[serial_mask] = pd.to_datetime(numeric[serial_mask], unit="D", origin="1899-12-30", errors="coerce")

    text = s.astype("string").str.strip()
    text = text.replace({"": pd.NA, "nan": pd.NA, "NaT": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    pending = out.isna() & text.notna()

    # Fechas anuales-mensuales como 2024M01, 2024-M01, 2024/01, 202401.
    compact = text.str.upper().str.replace(r"[\s._/\-]", "", regex=True)
    compact = compact.str.replace("M", "", regex=False)
    month_code = compact.str.fullmatch(r"(?:19|20)\d{2}(?:0[1-9]|1[0-2])", na=False) & pending
    if month_code.any():
        out.loc[month_code] = pd.to_datetime(compact[month_code], format="%Y%m", errors="coerce")

    # Fechas compactas YYYYMMDD.
    compact8 = text.str.replace(r"[^0-9]", "", regex=True)
    day_code = compact8.str.fullmatch(r"(?:19|20)\d{6}", na=False) & out.isna() & text.notna()
    if day_code.any():
        out.loc[day_code] = pd.to_datetime(compact8[day_code], format="%Y%m%d", errors="coerce")

    # Formatos mensuales MM/YYYY y MM-YYYY.
    mm_yyyy = text.str.fullmatch(r"(?:0?[1-9]|1[0-2])[/\-](?:19|20)\d{2}", na=False) & out.isna() & text.notna()
    if mm_yyyy.any():
        vals = text[mm_yyyy].str.replace("-", "/", regex=False)
        out.loc[mm_yyyy] = pd.to_datetime(vals, format="%m/%Y", errors="coerce")

    # Meses en español: "enero 2024", "ene-2024", "2024 enero".
    lowered = text.str.lower()
    for month_name, month_number in MONTHS_ES.items():
        lowered = lowered.str.replace(rf"\b{month_name}\b", f"{month_number:02d}", regex=True)
    lowered_compact = lowered.str.replace(r"[\s._/\-]", "", regex=True)
    lowered_compact = lowered_compact.str.replace("M", "", regex=False).str.replace("m", "", regex=False)
    m_y = lowered.str.fullmatch(r"(?:0?[1-9]|1[0-2])[\s/\-]+(?:19|20)\d{2}", na=False) & out.isna() & text.notna()
    if m_y.any():
        vals = lowered[m_y].str.replace(r"[\s/\-]+", " ", regex=True)
        out.loc[m_y] = pd.to_datetime(vals, format="%m %Y", errors="coerce")
    y_m = lowered.str.fullmatch(r"(?:19|20)\d{2}[\s/\-]+(?:0?[1-9]|1[0-2])", na=False) & out.isna() & text.notna()
    if y_m.any():
        vals = lowered[y_m].str.replace(r"[\s/\-]+", " ", regex=True)
        out.loc[y_m] = pd.to_datetime(vals, format="%Y %m", errors="coerce")

    # Para fechas con día explícito (p. ej. 31/01/2024 o ISO) usar parser mixto,
    # únicamente para las que aún no se reconocieron como mes/año.
    remaining = out.isna() & text.notna()
    if remaining.any():
        try:
            parsed = pd.to_datetime(text[remaining], errors="coerce", dayfirst=True, format="mixed")
        except TypeError:  # compatibilidad con pandas previos a format='mixed'
            parsed = pd.to_datetime(text[remaining], errors="coerce", dayfirst=True)
        out.loc[remaining] = parsed

    return pd.to_datetime(out, errors="coerce").dt.to_period("M").dt.to_timestamp()


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

    # En SIPSA-P, algunos años identifican la unidad directamente en el encabezado
    # de precio (p. ej. "Precio_por_kilogramo"), sin una columna de unidad separada.
    # Solo se infiere COP/kg cuando el encabezado lo declara explícitamente.
    price_header = norm_text(mapped["price"])
    inferred_unit = "COP/kg" if ("por kilogramo" in price_header or "por kilo" in price_header or re.search(r"\bkg\b", price_header)) else ""
    if mapped["unit"]:
        unit_values = table[mapped["unit"]].astype(str).str.strip()
        unit_values = unit_values.replace({"nan": "", "None": "", "<NA>": ""})
        if inferred_unit:
            unit_values = unit_values.where(unit_values.ne(""), inferred_unit)
    else:
        unit_values = pd.Series(inferred_unit, index=table.index, dtype="object")

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
    exts = {".csv", ".txt", ".xlsx", ".xls", ".dta"}
    for path in raw_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        if path.name.startswith("~$") or path.name == ".done":
            continue
        if path.suffix.lower() == ".dta":
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
                    frame = pd.read_csv(path, sep=None, engine="python", encoding=encoding)
                    if frame.shape[1] > 1:
                        break
                except Exception:
                    frame = None
            if frame is not None and not frame.empty:
                yield str(path), frame


def load_sipsa_data(raw_dir: str | Path, column_map: Optional[Dict[str, Optional[str]]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Lee archivos públicos descargados de SIPSA-P y devuelve filas canónicas + auditoría.

    No realiza imputación ni crea observaciones. COLUMN_MAP permite fijar los nombres exactos
    de las columnas cuando la estructura del ZIP cambia.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"No existe {raw_dir}. Descargue archivos SIPSA-P de DANE y ubíquelos allí.")
    _extract_archives(raw_dir)
    frames, audit = [], []
    for source, table in _iter_tables(raw_dir):
        canonical = _canonicalize_table(table, source, column_map)
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


def compute_irfs(results: Dict[str, Any], products: Sequence[str], W_dense: np.ndarray, horizon: int = 12) -> pd.DataFrame:
    """IRF de un choque climático exógeno de +1 DE, promediando todos los mercados de origen.

    Reporta respuesta directa (mercado afectado), spillover medio por mercado receptor,
    media de respuesta en la red y suma de respuesta en toda la red. Son IRF puntuales,
    sin intervalos de confianza; no deben leerse como causalidad sin identificar el choque.
    """
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


def residual_moran(results: Dict[str, Any], W, markets: Sequence[str]) -> pd.DataFrame:
    from esda.moran import Moran
    rows = []
    for product, fit in results.items():
        resid = fit.resids
        temp = resid.rename("residuo").reset_index()
        # PanelOLS suele devolver entidad y tiempo como las dos columnas del índice.
        entity_col, time_col = temp.columns[0], temp.columns[1]
        temp = temp.rename(columns={entity_col: "mercado_key", time_col: "fecha"})
        temp["fecha"] = pd.to_datetime(temp["fecha"])
        latest = temp["fecha"].max()
        values = temp[temp["fecha"] == latest].set_index("mercado_key")["residuo"].reindex(list(markets)).to_numpy()
        mi = Moran(values, W, permutations=999, two_tailed=True)
        rows.append({"producto_dependiente": product, "fecha_residuo": latest, "Moran_I_residuos": float(mi.I), "p_sim": float(mi.p_sim), "n": int(len(values))})
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
