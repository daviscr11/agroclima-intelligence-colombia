# -*- coding: utf-8 -*-
"""Pruebas unitarias de las reglas de integridad del JSON publicado."""
import unittest
from datetime import date

from validate_observatorio_integrity import collect_integrity_errors, month_range


class ObservatorioIntegrityTests(unittest.TestCase):
    def make_payload(self):
        months = month_range("2020-01", "2020-03")
        products = ["Tomate", "Pimenton"]
        series = []
        base = {"Tomate": [1000.0, 1100.0, 1050.0], "Pimenton": [2000.0, 2100.0, 2050.0]}
        for month in months:
            for product in products:
                price = base[product][months.index(month)]
                series.append({
                    "fecha": f"{month}-01T00:00:00",
                    "producto": product,
                    "precio_promedio_mercados": price,
                    "n_mercados": 2,
                })
        return {
            "status": "ready",
            "metadata": {
                "start_date": "2020-01", "end_date": "2020-03", "n_months": 3,
                "n_markets": 2, "n_model_observations": 2,
                "products": [{"name": p, "unit": "Kilogramo"} for p in products],
            },
            "price_series": series,
            "markets": [{"mercado": "A"}, {"mercado": "B"}],
        }

    def errors(self, payload):
        return collect_integrity_errors(payload, "2020-01", "2020-03")

    def test_valid_payload_passes(self):
        self.assertEqual(self.errors(self.make_payload()), [])

    def test_missing_month_fails(self):
        payload = self.make_payload()
        payload["price_series"] = [r for r in payload["price_series"] if not (r["producto"] == "Tomate" and r["fecha"].startswith("2020-02"))]
        self.assertTrue(any("Falta observación mensual" in e for e in self.errors(payload)))

    def test_nonpositive_price_fails(self):
        payload = self.make_payload()
        payload["price_series"][0]["precio_promedio_mercados"] = 0
        self.assertTrue(any("Precio no positivo" in e for e in self.errors(payload)))

    def test_duplicate_month_product_fails(self):
        payload = self.make_payload()
        payload["price_series"].append(dict(payload["price_series"][0]))
        self.assertTrue(any("Registro duplicado" in e for e in self.errors(payload)))

    def test_scale_jump_fails(self):
        payload = self.make_payload()
        next(row for row in payload["price_series"] if row["producto"] == "Tomate" and row["fecha"].startswith("2020-02"))["precio_promedio_mercados"] = 1.1
        self.assertTrue(any("Posible cambio anómalo de escala" in e for e in self.errors(payload)))

    def test_wrong_coverage_fails(self):
        payload = self.make_payload()
        payload["price_series"][0]["n_mercados"] = 1
        self.assertTrue(any("Cobertura desigual" in e for e in self.errors(payload)))


if __name__ == "__main__":
    unittest.main()
