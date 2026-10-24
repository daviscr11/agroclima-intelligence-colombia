import sys
from pathlib import Path
import unittest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agroclima_spvarx_real import _parse_date_column, _canonicalize_table


class SipsaDateParserTests(unittest.TestCase):
    def test_spanish_month_year_labels(self):
        source = pd.Series(["ene-22", "feb-23", "abr-24", "dic-24", "set-24", "septiembre-2024"])
        parsed = _parse_date_column(source)
        expected = pd.to_datetime(["2022-01-01", "2023-02-01", "2024-04-01", "2024-12-01", "2024-09-01", "2024-09-01"])
        self.assertTrue(parsed.equals(pd.Series(expected)))

    def test_legacy_date_formats_remain_supported(self):
        source = pd.Series(["201901", "202401", "15/02/2024", "2024-03-15", "01-24"])
        parsed = _parse_date_column(source)
        expected = pd.to_datetime(["2019-01-01", "2024-01-01", "2024-02-01", "2024-03-01", "2024-01-01"])
        self.assertTrue(parsed.equals(pd.Series(expected)))

    def test_official_price_per_kilogram_schema_gets_documented_unit(self):
        table = pd.DataFrame({
            "Fecha": ["ene-22", "dic-24"],
            "Grupo": ["Frutas", "Frutas"],
            "Producto": ["Tomate de árbol", "Tomate de árbol"],
            "Mercado": ["Corabastos", "Corabastos"],
            "Precio_promedio_por_kilogramo": [3500, 4200],
        })
        got = _canonicalize_table(table, "BaseDatos-SIPSA_P-Mensual-2024.xlsx")
        self.assertIsNotNone(got)
        self.assertEqual(got["fecha"].dt.strftime("%Y-%m").tolist(), ["2022-01", "2024-12"])
        self.assertEqual(got["unidad"].tolist(), ["Kilogramo", "Kilogramo"])
        self.assertEqual(got["precio"].tolist(), [3500, 4200])

    def test_no_unit_is_invented_when_price_field_does_not_state_per_kg(self):
        table = pd.DataFrame({
            "Fecha": ["ene-24"], "Producto": ["Pimentón"], "Mercado": ["Corabastos"],
            "Precio promedio": [3000],
        })
        got = _canonicalize_table(table, "sample.xlsx")
        self.assertIsNotNone(got)
        self.assertEqual(got["unidad"].iloc[0], "")

    def test_2019_official_fuente_field_is_used_as_market_identifier(self):
        # DANE's 2019 SIPSA-P file calls the market/source field "Fuente".
        table = pd.DataFrame({
            "Fecha": ["2019-01-01", "2019-02-01"],
            "Grupo": ["Frutas", "Frutas"],
            "Producto": ["Tomate de árbol", "Tomate de árbol"],
            "Fuente": ["Armenia, Mercar", "Cartagena, Bazurto"],
            "Precio_por_kilogramo": [3500, 3600],
        })
        got = _canonicalize_table(table, "Mensual_2019.csv")
        self.assertIsNotNone(got)
        self.assertEqual(len(got), 2)
        self.assertEqual(got["mercado"].tolist(), ["Armenia, Mercar", "Cartagena, Bazurto"])
        self.assertEqual(got["fecha"].dt.strftime("%Y-%m").tolist(), ["2019-01", "2019-02"])
        self.assertEqual(got["unidad"].tolist(), ["Kilogramo", "Kilogramo"])


if __name__ == "__main__":
    unittest.main()
