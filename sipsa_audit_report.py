# -*- coding: utf-8 -*-
"""Genera informes SIPSA-P mientras el lector principal procesa cada tabla.

Distingue filas físicas de origen, registros canónicos antes de deduplicar, duplicados exactos
eliminados y observaciones finales. No aplica un multiplicador monetario global; registra los
patrones de separadores que se normalizan y qué formatos se omiten por ser exportaciones gemelas.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agroclima_spvarx_real import (
    MONTHS_ES,
    _parse_date_column,
    _resolve_col,
    norm_text,
)

PRICE_GROUP_DOT = re.compile(r"[+-]?\d{1,3}(?:\.\d{3})+")
PRICE_GROUP_COMMA = re.compile(r"[+-]?\d{1,3}(?:,\d{3})+")
SCALE_COUNT_COLUMNS = [
    "tokens_punto_miles_interpretados",
    "tokens_coma_miles_interpretados",
    "tokens_formato_mixto_normalizados",
    "valores_precio_nativos_numericos",
    "tokens_punto_decimal_no_miles",
    "tokens_precio_no_nulo",
]


def _date_series_from_table(table: pd.DataFrame, source: str) -> pd.Series:
    """Resuelve fechas igual que el parser principal, incluidos meses en español."""
    mapped = {role: _resolve_col(table.columns, role) for role in ["date", "year", "month"]}
    if mapped["date"]:
        return _parse_date_column(table[mapped["date"]])
    if mapped["year"] and mapped["month"]:
        year_source = table[mapped["year"]]
    elif mapped["month"]:
        year_match = re.search(r"20\d{2}", Path(source.split("::")[0]).name)
        if not year_match:
            return pd.Series(pd.NaT, index=table.index, dtype="datetime64[ns]")
        year_source = pd.Series(int(year_match.group(0)), index=table.index)
    else:
        return pd.Series(pd.NaT, index=table.index, dtype="datetime64[ns]")

    years = pd.to_numeric(year_source, errors="coerce")
    month_raw = table[mapped["month"]].astype(str).str.strip().str.lower()
    month_map = {norm_text(k): v for k, v in MONTHS_ES.items()}
    month_text = month_raw.map(lambda value: month_map.get(norm_text(value)))
    months = pd.to_numeric(month_raw, errors="coerce").fillna(month_text)
    months = pd.to_numeric(months, errors="coerce")
    result = pd.to_datetime({"year": years, "month": months, "day": 1}, errors="coerce")
    return pd.to_datetime(result, errors="coerce").dt.to_period("M").dt.to_timestamp()


def _product_series(table: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    col = _resolve_col(table.columns, "product")
    if not col:
        blank = pd.Series("", index=table.index, dtype="object")
        return blank, blank
    labels = table[col].fillna("").astype(str).str.strip()
    keys = labels.map(norm_text)
    invalid = keys.isin({"", "nan", "none", "null", "na"})
    return labels.mask(invalid, ""), keys.mask(invalid, "")


def _scale_patterns(table: pd.DataFrame) -> dict[str, int]:
    result = {name: 0 for name in SCALE_COUNT_COLUMNS}
    price_col = _resolve_col(table.columns, "price")
    if not price_col:
        return result
    values = table[price_col]
    if pd.api.types.is_numeric_dtype(values):
        result["valores_precio_nativos_numericos"] = int(values.notna().sum())
        result["tokens_precio_no_nulo"] = int(values.notna().sum())
        return result
    text = values.fillna("").astype(str).str.strip()
    meaningful = text.str.lower().ne("") & ~text.str.lower().isin({"nan", "none", "null", "na", "<na>"})
    result["tokens_precio_no_nulo"] = int(meaningful.sum())
    text = text[meaningful]
    has_dot = text.str.contains(r"\.", regex=True)
    has_comma = text.str.contains(",", regex=False)
    dot_group = text.str.fullmatch(PRICE_GROUP_DOT).fillna(False)
    comma_group = text.str.fullmatch(PRICE_GROUP_COMMA).fillna(False)
    result["tokens_punto_miles_interpretados"] = int(dot_group.sum())
    result["tokens_coma_miles_interpretados"] = int(comma_group.sum())
    result["tokens_formato_mixto_normalizados"] = int((has_dot & has_comma).sum())
    result["tokens_punto_decimal_no_miles"] = int((has_dot & ~has_comma & ~dot_group).sum())
    return result


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def assert_selected_products_scale(report_path: str | Path, selected_products: list[str], min_ratio: float = 0.1, max_ratio: float = 10.0) -> None:
    """Bloquea publicación si la escala mediana interanual de un producto elegido salta >10x.

    La alerta general se conserva en el informe para todos los productos. Solo los productos
    elegidos para el modelo bloquean, evitando falsos positivos de categorías con unidades distintas.
    """
    report = pd.read_csv(report_path)
    report["producto_key"] = report["producto"].map(norm_text)
    selected_keys = {norm_text(product) for product in selected_products}
    subset = report[report["producto_key"].isin(selected_keys)].copy()
    if subset.empty:
        raise RuntimeError(f"El informe SIPSA no contiene los productos seleccionados: {selected_products}")
    ratio = pd.to_numeric(subset["razon_mediana_vs_anio_anterior"], errors="coerce")
    bad = subset[ratio.notna() & ((ratio < min_ratio) | (ratio > max_ratio))]
    if not bad.empty:
        details = bad[["anio", "producto", "mediana_anio_anterior", "precio_mediano_canonico", "razon_mediana_vs_anio_anterior"]].to_dict(orient="records")
        raise RuntimeError(
            "Control de escala interanual falló para un producto seleccionado; no se publicará el modelo. "
            f"Razones fuera de [{min_ratio:g}, {max_ratio:g}]: {details[:12]}"
        )


class SipsaAuditCollector:
    """Acumula diagnósticos desde la misma pasada de lectura del parser SIPSA-P."""

    def __init__(self, raw_dir: str | Path, start_year: int, end_year: int, min_scale_ratio: float = 0.1, max_scale_ratio: float = 10.0):
        self.raw_dir = Path(raw_dir)
        self.start_year = int(start_year)
        self.end_year = int(end_year)
        self.min_scale_ratio = float(min_scale_ratio)
        self.max_scale_ratio = float(max_scale_ratio)
        if not (0 < self.min_scale_ratio < self.max_scale_ratio):
            raise ValueError("Los umbrales de escala deben cumplir 0 < mínimo < máximo.")
        self.file_rows: list[dict[str, Any]] = []
        self.original_groups: list[pd.DataFrame] = []
        self.canonical_frames: list[pd.DataFrame] = []
        self.scale_groups: list[pd.DataFrame] = []
        self.total_unassigned_original_rows = 0

    def capture_table(self, source: str, table: pd.DataFrame, canonical: pd.DataFrame | None) -> None:
        """Callback compatible con `load_sipsa_data(table_audit_callback=...)`."""
        dates = _date_series_from_table(table, source)
        labels, keys = _product_series(table)
        valid = dates.notna() & keys.ne("")
        valid_period = valid & dates.dt.year.between(self.start_year, self.end_year)
        self.total_unassigned_original_rows += int((~valid).sum())
        if valid_period.any():
            original = pd.DataFrame({
                "anio": dates.loc[valid_period].dt.year.astype(int),
                "producto_key": keys.loc[valid_period].astype(str),
                "producto": labels.loc[valid_period].astype(str),
            })
            original = original.groupby(["anio", "producto_key", "producto"], as_index=False, dropna=False).size()
            original = original.rename(columns={"size": "filas_originales"})
            self.original_groups.append(original)

            price_col = _resolve_col(table.columns, "price")
            if price_col:
                values = table[price_col]
                text = values.fillna("").astype(str).str.strip()
                has_dot = text.str.contains(r"\.", regex=True)
                has_comma = text.str.contains(",", regex=False)
                dot_group = text.str.fullmatch(PRICE_GROUP_DOT).fillna(False)
                comma_group = text.str.fullmatch(PRICE_GROUP_COMMA).fillna(False)
                patterns = pd.DataFrame({
                    "anio": dates.loc[valid_period].dt.year.astype(int),
                    "producto_key": keys.loc[valid_period].astype(str),
                    "tokens_punto_miles_interpretados": dot_group.loc[valid_period].astype(int).to_numpy(),
                    "tokens_coma_miles_interpretados": comma_group.loc[valid_period].astype(int).to_numpy(),
                    "tokens_formato_mixto_normalizados": (has_dot & has_comma).loc[valid_period].astype(int).to_numpy(),
                    "valores_precio_nativos_numericos": (values.notna().loc[valid_period].astype(int).to_numpy() if pd.api.types.is_numeric_dtype(values) else np.zeros(int(valid_period.sum()), dtype=int)),
                    "tokens_punto_decimal_no_miles": (has_dot & ~has_comma & ~dot_group).loc[valid_period].astype(int).to_numpy(),
                    "tokens_precio_no_nulo": (values.notna() & text.str.lower().ne("") & ~text.str.lower().isin({"nan", "none", "null", "na", "<na>"})).loc[valid_period].astype(int).to_numpy(),
                })
                self.scale_groups.append(patterns.groupby(["anio", "producto_key"], as_index=False).sum(numeric_only=True))

        pattern_counts = _scale_patterns(table)
        canonical_count = 0 if canonical is None or canonical.empty else len(canonical)
        if canonical is not None and not canonical.empty:
            can = canonical.copy()
            can["fecha"] = pd.to_datetime(can["fecha"], errors="coerce")
            can = can[can["fecha"].dt.year.between(self.start_year, self.end_year)].copy()
            can["anio"] = can["fecha"].dt.year.astype(int)
            can["producto_key"] = can["producto"].map(norm_text)
            can["mercado_key"] = can["mercado"].map(norm_text)
            can["unidad_key"] = can["unidad"].map(norm_text)
            can["variedad_key"] = can["variedad"].map(norm_text)
            self.canonical_frames.append(can)
            min_date, max_date = can["fecha"].min(), can["fecha"].max()
            market_count, product_count = int(can["mercado_key"].nunique()), int(can["producto_key"].nunique())
        else:
            min_date = max_date = pd.NaT
            market_count = product_count = 0

        self.file_rows.append({
            "archivo_hoja": source,
            "formato": Path(source.split("::")[0]).suffix.lower(),
            "filas_originales_tabla": int(len(table)),
            "filas_canonicas_reconocidas": int(canonical_count),
            "filas_no_canonizadas": int(max(0, len(table) - canonical_count)),
            "mercados_reconocidos": market_count,
            "productos_reconocidos": product_count,
            "fecha_min_canonica": min_date.strftime("%Y-%m") if pd.notna(min_date) else "",
            "fecha_max_canonica": max_date.strftime("%Y-%m") if pd.notna(max_date) else "",
            "estado": "reconocido" if canonical_count else "esquema_no_reconocido_o_sin_precios_validos",
            **pattern_counts,
            "tokens_precio_no_nulo": int(table[_resolve_col(table.columns, "price")].notna().sum()) if _resolve_col(table.columns, "price") else 0,
            "columnas_detectadas": " | ".join(map(str, list(table.columns)[:50])),
        })

    def write(self, canonical_raw: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
        if not self.canonical_frames:
            raise RuntimeError("La auditoría no reconoció ninguna tabla SIPSA canónica.")
        before_dedup = pd.concat(self.canonical_frames, ignore_index=True)
        keys = ["fecha", "mercado_key", "producto_key", "precio", "unidad_key", "variedad_key"]
        if any(key not in before_dedup for key in keys):
            raise RuntimeError("Faltan claves necesarias para reproducir la deduplicación del lector SIPSA.")

        duplicate_mask = before_dedup.duplicated(subset=keys, keep="first")
        duplicate_rows = before_dedup.loc[duplicate_mask]
        before_counts = before_dedup.groupby(["anio", "producto_key"], as_index=False).agg(
            filas_canonicas_antes_deduplicar=("precio", "size"), producto=("producto", "first")
        )
        duplicate_counts = duplicate_rows.groupby(["anio", "producto_key"], as_index=False).size().rename(
            columns={"size": "duplicados_exactos_eliminados"}
        )

        final = canonical_raw.copy()
        final["fecha"] = pd.to_datetime(final["fecha"], errors="coerce")
        final = final[final["fecha"].dt.year.between(self.start_year, self.end_year)].copy()
        final["anio"] = final["fecha"].dt.year.astype(int)
        final["producto_key"] = final["producto"].map(norm_text)
        final_counts = final.groupby(["anio", "producto_key"], as_index=False).agg(
            observaciones_reconocidas_tras_deduplicar=("precio", "size"), producto=("producto", "first")
        )

        if self.original_groups:
            originals = pd.concat(self.original_groups, ignore_index=True)
            originals = originals.groupby(["anio", "producto_key"], as_index=False).agg(
                filas_originales=("filas_originales", "sum"), producto=("producto", "first")
            )
        else:
            originals = pd.DataFrame(columns=["anio", "producto_key", "filas_originales", "producto"])

        key_frames = [frame[["anio", "producto_key"]] for frame in [before_counts, originals, duplicate_counts, final_counts]]
        all_keys = pd.concat(key_frames, ignore_index=True).drop_duplicates()
        labels = pd.concat([
            before_counts[["anio", "producto_key", "producto"]],
            originals[["anio", "producto_key", "producto"]],
            final_counts[["anio", "producto_key", "producto"]],
        ], ignore_index=True).dropna(subset=["producto"]).drop_duplicates(["anio", "producto_key"], keep="first")
        report = all_keys.merge(labels, on=["anio", "producto_key"], how="left")
        report = report.merge(originals.drop(columns=["producto"], errors="ignore"), on=["anio", "producto_key"], how="left")
        report = report.merge(before_counts.drop(columns=["producto"], errors="ignore"), on=["anio", "producto_key"], how="left")
        report = report.merge(duplicate_counts, on=["anio", "producto_key"], how="left")
        report = report.merge(final_counts.drop(columns=["producto"], errors="ignore"), on=["anio", "producto_key"], how="left")

        if self.scale_groups:
            scale_df = pd.concat(self.scale_groups, ignore_index=True).groupby(["anio", "producto_key"], as_index=False).sum(numeric_only=True)
            report = report.merge(scale_df, on=["anio", "producto_key"], how="left")
        count_cols = ["filas_originales", "filas_canonicas_antes_deduplicar", "duplicados_exactos_eliminados", "observaciones_reconocidas_tras_deduplicar", *SCALE_COUNT_COLUMNS]
        for col in count_cols:
            if col not in report:
                report[col] = 0
            report[col] = pd.to_numeric(report[col], errors="coerce").fillna(0).astype(int)
        report["filas_descartadas_por_filtro_canonico"] = (report["filas_originales"] - report["filas_canonicas_antes_deduplicar"]).clip(lower=0)
        report["reconciliacion_filas"] = (
            report["filas_canonicas_antes_deduplicar"] - report["duplicados_exactos_eliminados"]
            == report["observaciones_reconocidas_tras_deduplicar"]
        )
        # Resume escala anual de precios ya deduplicados y marca cambios interanuales que merecen revisión.
        price_stats = final.groupby(["anio", "producto_key"], as_index=False).agg(
            producto_stats=("producto", "first"),
            precio_minimo_canonico=("precio", "min"),
            precio_mediano_canonico=("precio", "median"),
            precio_maximo_canonico=("precio", "max"),
        )
        price_stats = price_stats.sort_values(["producto_key", "anio"]).reset_index(drop=True)
        price_stats["mediana_anio_anterior"] = price_stats.groupby("producto_key")["precio_mediano_canonico"].shift(1)
        price_stats["anio_anterior"] = price_stats.groupby("producto_key")["anio"].shift(1)
        consecutive = price_stats["anio"].sub(price_stats["anio_anterior"]).eq(1)
        price_stats["razon_mediana_vs_anio_anterior"] = (price_stats["precio_mediano_canonico"] / price_stats["mediana_anio_anterior"]).where(consecutive)
        # The default thresholds match the JSON integrity guard; alerts are diagnostic until
        # we know which product/unit combinations are actually selected for the model.
        ratio = price_stats["razon_mediana_vs_anio_anterior"]
        price_stats["control_escala_interanual"] = np.where(
            ~consecutive, "SIN_ANIO_CONSECUTIVO_ANTERIOR",
            np.where((ratio < self.min_scale_ratio) | (ratio > self.max_scale_ratio), "REVISAR_SALTO_ESCALA", "OK")
        )
        report = report.merge(
            price_stats.drop(columns=["producto_stats", "anio_anterior"], errors="ignore"),
            on=["anio", "producto_key"], how="left",
        )
        report["regla_de_escala"] = (
            "Separador agrupado en 3 dígitos se interpreta como miles; formatos mixtos se normalizan; "
            "sin factor monetario global; CSV/TXT prevalece sobre .dta equivalente; mediana anual comparada con año previo"
        )
        report = report.sort_values(["anio", "producto_key"]).reset_index(drop=True)

        mismatches = report[~report["reconciliacion_filas"]]
        if not mismatches.empty:
            detail = mismatches[["anio", "producto", "filas_canonicas_antes_deduplicar", "duplicados_exactos_eliminados", "observaciones_reconocidas_tras_deduplicar"]].to_dict(orient="records")
            raise RuntimeError(f"La auditoría SIPSA no reconcilia con el lector principal: {detail[:12]}")
        if (report["filas_originales"] < report["filas_canonicas_antes_deduplicar"]).any():
            detail = report.loc[report["filas_originales"] < report["filas_canonicas_antes_deduplicar"], ["anio", "producto", "filas_originales", "filas_canonicas_antes_deduplicar"]].to_dict(orient="records")
            raise RuntimeError(f"Más filas canónicas que filas de origen por año/producto: {detail[:12]}")

        files_df = pd.DataFrame(self.file_rows)
        processed_paths = {str(Path(str(row["archivo_hoja"]).split("::")[0]).resolve()) for row in self.file_rows}
        # Registra archivos Stata omitidos por contar con CSV/TXT equivalente; no los vuelve a leer.
        for path in self.raw_dir.rglob("*.dta"):
            siblings = [sibling for sibling in path.parent.iterdir() if sibling.is_file() and sibling.suffix.lower() in {".csv", ".txt"} and norm_text(sibling.stem) == norm_text(path.stem)]
            if siblings and str(path.resolve()) not in processed_paths:
                files_df.loc[len(files_df)] = {
                    "archivo_hoja": str(path), "formato": ".dta", "filas_originales_tabla": np.nan,
                    "filas_canonicas_reconocidas": 0, "filas_no_canonizadas": np.nan,
                    "mercados_reconocidos": 0, "productos_reconocidos": 0,
                    "fecha_min_canonica": "", "fecha_max_canonica": "",
                    "estado": "omitido_exportacion_stata_equivalente",
                    **{name: 0 for name in SCALE_COUNT_COLUMNS},
                    "columnas_detectadas": f"Omitido; CSV/TXT equivalente preferido: {', '.join(s.name for s in siblings)}",
                }

        original_rows_total = int(sum(row["filas_originales_tabla"] for row in self.file_rows))
        canonical_before_total = int(len(before_dedup))
        duplicate_total = int(duplicate_mask.sum())
        final_total = int(len(final))
        rows_by_year = final.groupby("anio").size().rename("observaciones_despues_deduplicar").reset_index().to_dict(orient="records")
        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "period_start_year": self.start_year,
            "period_end_year": self.end_year,
            "source_tables_processed": int(len(self.file_rows)),
            "source_tables_recognized": int(sum(row["estado"] == "reconocido" for row in self.file_rows)),
            "source_tables_unrecognized": int(sum(row["estado"] != "reconocido" for row in self.file_rows)),
            "source_tables_stata_omitted_in_favor_of_text": int((files_df["estado"] == "omitido_exportacion_stata_equivalente").sum()),
            "original_rows_in_processed_tables": original_rows_total,
            "original_rows_without_parseable_year_product": int(self.total_unassigned_original_rows),
            "canonical_rows_before_exact_deduplication": canonical_before_total,
            "exact_duplicate_rows_removed": duplicate_total,
            "canonical_rows_after_exact_deduplication": final_total,
            "year_product_rows": int(len(report)),
            "reconciliation_passed": bool(report["reconciliacion_filas"].all()),
            "scale_controls_total": {col: int(report[col].sum()) for col in SCALE_COUNT_COLUMNS},
            "interannual_scale_alerts": int(report["control_escala_interanual"].eq("REVISAR_SALTO_ESCALA").sum()),
            "min_interannual_scale_ratio": self.min_scale_ratio,
            "max_interannual_scale_ratio": self.max_scale_ratio,
            "rows_by_year_after_deduplication": rows_by_year,
            "notes": [
                "Filas originales por año/producto cuentan filas fuente con fecha y producto parseables, antes de filtrar precio o mercado.",
                "Filas canónicas reconocidas son los registros con fecha, precio positivo, mercado y producto válidos antes de deduplicar.",
                "Duplicados eliminados reproducen las claves de load_sipsa_data: fecha, mercado, producto, precio, unidad y variedad.",
                "Los conteos de separadores muestran patrones de entrada; no se aplica un factor monetario global.",
                "Las exportaciones .dta con CSV/TXT equivalente se registran como omitidas para evitar mezclar representaciones de escala.",
                "El total original cubre las tablas leídas por el parser; no suma las filas desconocidas de .dta omitidos, pero estos quedan registrados por separado.",
                "Las filas fuente no asignables a año/producto permanecen contabilizadas en el resumen y en el informe por archivo.",
            ],
        }
        output_dir = Path(output_dir)
        paths = {
            "files_csv": output_dir / "sipsa_audit_files.csv",
            "year_product_csv": output_dir / "sipsa_audit_year_product.csv",
            "summary_json": output_dir / "sipsa_audit_summary.json",
        }
        _atomic_csv(files_df, paths["files_csv"])
        _atomic_csv(report, paths["year_product_csv"])
        _atomic_json(summary, paths["summary_json"])

        print("[SIPSA auditoría] Informes detallados generados:")
        print(f" - {paths['files_csv']}")
        print(f" - {paths['year_product_csv']} ({len(report)} combinaciones año-producto)")
        print(f" - {paths['summary_json']}")
        print(f" - Tablas procesadas: {summary['source_tables_recognized']} reconocidas; {summary['source_tables_unrecognized']} sin esquema canónico; {summary['source_tables_stata_omitted_in_favor_of_text']} .dta equivalentes omitidas.")
        print(f" - Filas canónicas: {canonical_before_total} antes de deduplicar; {duplicate_total} duplicados exactos eliminados; {final_total} después.")
        return paths
