# -*- coding: utf-8 -*-
"""Pruebas de la trazabilidad SIPSA-P por archivo, año y producto."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from agroclima_spvarx_real import load_sipsa_data
from sipsa_audit_report import SipsaAuditCollector, assert_selected_products_scale


class SipsaAuditReportTests(unittest.TestCase):
    def test_report_tracks_rows_duplicates_and_scale_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            out_dir = root / "data"
            raw_dir.mkdir()
            csv_text = (
                "Fecha,Fuente,Producto,Precio,Unidad\n"
                "ene-24,Mercado A,Tomate de arbol,1.234,Kilogramo\n"
                "feb-24,Mercado A,Tomate de arbol,1.250,Kilogramo\n"
            )
            (raw_dir / "mensual_a.csv").write_text(csv_text, encoding="utf-8")
            (raw_dir / "mensual_b.csv").write_text(csv_text, encoding="utf-8")

            collector = SipsaAuditCollector(raw_dir, 2024, 2024)
            canonical, _ = load_sipsa_data(raw_dir, table_audit_callback=collector.capture_table)
            canonical["fecha"] = pd.to_datetime(canonical["fecha"])
            canonical = canonical[canonical["fecha"].dt.year.eq(2024)].copy()
            paths = collector.write(canonical, out_dir)

            report = pd.read_csv(paths["year_product_csv"])
            row = report.loc[report["producto_key"].eq("tomate de arbol")].iloc[0]
            self.assertEqual(int(row["filas_originales"]), 4)
            self.assertEqual(int(row["filas_canonicas_antes_deduplicar"]), 4)
            self.assertEqual(int(row["duplicados_exactos_eliminados"]), 2)
            self.assertEqual(int(row["observaciones_reconocidas_tras_deduplicar"]), 2)
            self.assertEqual(int(row["tokens_punto_miles_interpretados"]), 4)
            self.assertTrue(bool(row["reconciliacion_filas"]))

            files = pd.read_csv(paths["files_csv"])
            self.assertEqual(int(files["filas_originales_tabla"].sum()), 4)


    def test_selected_product_interannual_scale_jump_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            out_dir = root / "data"
            raw_dir.mkdir()
            (raw_dir / "monthly.csv").write_text(
                "Fecha,Fuente,Producto,Precio,Unidad\n"
                "ene-23,Mercado A,Tomate de arbol,1000,Kilogramo\n"
                "ene-24,Mercado A,Tomate de arbol,1,Kilogramo\n",
                encoding="utf-8",
            )
            collector = SipsaAuditCollector(raw_dir, 2023, 2024)
            canonical, _ = load_sipsa_data(raw_dir, table_audit_callback=collector.capture_table)
            paths = collector.write(canonical, out_dir)
            report = pd.read_csv(paths["year_product_csv"])
            self.assertIn("REVISAR_SALTO_ESCALA", report["control_escala_interanual"].tolist())
            with self.assertRaises(RuntimeError):
                assert_selected_products_scale(paths["year_product_csv"], ["Tomate de arbol"])

    def test_report_fails_if_postdedup_does_not_reconcile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            out_dir = root / "data"
            raw_dir.mkdir()
            (raw_dir / "mensual.csv").write_text(
                "Fecha,Fuente,Producto,Precio,Unidad\n"
                "ene-24,Mercado A,Tomate de arbol,1.234,Kilogramo\n",
                encoding="utf-8",
            )
            collector = SipsaAuditCollector(raw_dir, 2024, 2024)
            canonical, _ = load_sipsa_data(raw_dir, table_audit_callback=collector.capture_table)
            canonical = canonical.iloc[0:0].copy()
            with self.assertRaises(RuntimeError):
                collector.write(canonical, out_dir)


if __name__ == "__main__":
    unittest.main()
