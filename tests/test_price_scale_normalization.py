import tempfile
import unittest
from pathlib import Path

import pandas as pd

from agroclima_spvarx_real import _canonicalize_table, _iter_tables, _parse_number, load_sipsa_data


class SipsaPriceParsingTests(unittest.TestCase):
    def test_dane_dot_thousands_are_parsed_as_pesos(self):
        raw = pd.Series(["2.947", "1.150", "4.000", "560", "12.345.678"])
        result = _parse_number(raw).tolist()
        self.assertEqual(result, [2947.0, 1150.0, 4000.0, 560.0, 12345678.0])

    def test_decimal_and_mixed_locale_formats_are_preserved(self):
        raw = pd.Series(["2,5", "1.234,56", "1,234.56", "12.5"])
        result = _parse_number(raw).tolist()
        self.assertEqual(result, [2.5, 1234.56, 1234.56, 12.5])

    def test_csv_ingestion_preserves_price_tokens_before_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "SIPSA_P-Mensual-2022.csv"
            path.write_text(
                "Fecha;Mercado;Producto;Precio_promedio_por_kilogramo\n"
                "ene-22;Bogotá, D.C., Corabastos;Tomate de árbol;2.947\n"
                "ene-22;Arauca (Arauca);Tomate de árbol;560\n",
                encoding="utf-8",
            )
            tables = list(_iter_tables(Path(tmp)))
            self.assertEqual(len(tables), 1)
            source, frame = tables[0]
            self.assertEqual(str(frame["Precio_promedio_por_kilogramo"].iloc[0]), "2.947")
            canonical = _canonicalize_table(frame, source)
            self.assertIsNotNone(canonical)
            prices = dict(zip(canonical["mercado"], canonical["precio"]))
            self.assertEqual(prices["Bogotá, D.C., Corabastos"], 2947.0)
            self.assertEqual(prices["Arauca (Arauca)"], 560.0)

    def test_canonicalize_real_sipsa_field(self):
        frame = pd.DataFrame({
            "Fecha": ["ene-22", "ene-22"],
            "Mercado": ["Bogotá, D.C., Corabastos", "Arauca (Arauca)"],
            "Producto": ["Tomate de árbol", "Tomate de árbol"],
            "Precio_promedio_por_kilogramo": ["2.947", "560"],
        })
        result = _canonicalize_table(frame, "SIPSA_P-Mensual-2022/Mensual_22.dta")
        self.assertIsNotNone(result)
        self.assertEqual(result["precio"].tolist(), [2947.0, 560.0])
        self.assertEqual(result["unidad"].tolist(), ["Kilogramo", "Kilogramo"])


    def test_equivalent_csv_takes_priority_over_duplicate_stata_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "mensual 24.csv"
            dta_path = root / "Mensual_24.dta"
            csv_path.write_text(
                "Fecha;Mercado;Producto;Precio promedio por kilogramo*\n"
                "ene-24;Arauca (Arauca);Tomate de árbol;2.733\n",
                encoding="utf-8",
            )
            # The DTA representation has the same row but stores the textual token as a float.
            # This models the duplicate export in the official 2024 ZIP.
            pd.DataFrame({
                "Fecha": ["ene-24"],
                "Mercado": ["Arauca (Arauca)"],
                "Producto": ["Tomate de árbol"],
                "Precio_promedio_por_kilogramo_": [2.733],
            }).to_stata(dta_path, write_index=False, version=118)

            data, audit = load_sipsa_data(root)
            matched = data[(data["producto"] == "Tomate de árbol") & (data["mercado"] == "Arauca (Arauca)")]
            self.assertEqual(len(matched), 1)
            self.assertEqual(float(matched["precio"].iloc[0]), 2733.0)
            self.assertTrue(matched["fuente_archivo"].iloc[0].endswith("mensual 24.csv"))



if __name__ == "__main__":
    unittest.main()
