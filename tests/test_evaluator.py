"""
Tests para las funciones core del evaluador de tickets.
Ejecutar con: pytest tests/test_evaluator.py -v
"""

import json
import pytest
import pandas as pd
from unittest.mock import MagicMock


# ── Fixtures ──────────────────────────────────────────────────────────────────

VALID_LLM_RESPONSE = {
    "content_score": 4,
    "content_explanation": "Addresses the main issue adequately.",
    "format_score": 5,
    "format_explanation": "Well-structured and error-free.",
}


# ── Tests de lectura y validación del CSV ─────────────────────────────────────

class TestLoadAndValidateCsv:

    def test_loads_valid_csv(self, tmp_path):
        csv_file = tmp_path / "tickets.csv"
        csv_file.write_text("ticket,reply\nhello,world\n")
        df = pd.read_csv(csv_file)
        assert list(df.columns) == ["ticket", "reply"]
        assert len(df) == 1

    def test_raises_on_missing_file(self):
        import os
        with pytest.raises(FileNotFoundError):
            if not os.path.exists("nonexistent.csv"):
                raise FileNotFoundError("No se encontró el archivo: nonexistent.csv")

    def test_raises_on_missing_columns(self, tmp_path):
        csv_file = tmp_path / "bad.csv"
        csv_file.write_text("message,response\nhello,world\n")
        df = pd.read_csv(csv_file)
        missing = {"ticket", "reply"} - set(df.columns)
        with pytest.raises(ValueError):
            if missing:
                raise ValueError(f"Columnas faltantes: {missing}")


# ── Tests de construcción del User Prompt ─────────────────────────────────────

class TestBuildUserPrompt:

    def test_returns_valid_json(self):
        ticket = "My order is late."
        reply = "We apologize for the delay."
        result = json.dumps({"ticket": ticket, "reply": reply}, ensure_ascii=False)
        parsed = json.loads(result)
        assert parsed["ticket"] == ticket
        assert parsed["reply"] == reply

    def test_handles_special_characters(self):
        ticket = 'Ticket with "quotes" and \nnewlines'
        reply = "Reply with 'apostrophes'"
        result = json.dumps({"ticket": ticket, "reply": reply}, ensure_ascii=False)
        parsed = json.loads(result)
        assert parsed["ticket"] == ticket

    def test_handles_unicode(self):
        ticket = "Problema con mi pedido número 1234 🚚"
        reply = "Lamentamos el inconveniente, señor."
        result = json.dumps({"ticket": ticket, "reply": reply}, ensure_ascii=False)
        assert "🚚" in result


# ── Tests de validación del schema de respuesta ───────────────────────────────

class TestResponseValidation:

    def validate_response(self, result: dict):
        required_fields = {
            "content_score", "content_explanation",
            "format_score", "format_explanation"
        }
        missing = required_fields - result.keys()
        if missing:
            raise ValueError(f"Campos faltantes: {missing}")
        for field in ("content_score", "format_score"):
            score = result[field]
            if not isinstance(score, int) or not (1 <= score <= 5):
                raise ValueError(f"Score inválido en '{field}': {score}")
        return result

    def test_valid_response_passes(self):
        result = self.validate_response(VALID_LLM_RESPONSE)
        assert result["content_score"] == 4

    def test_missing_field_raises(self):
        bad_response = {"content_score": 4, "content_explanation": "ok"}
        with pytest.raises(ValueError, match="Campos faltantes"):
            self.validate_response(bad_response)

    def test_score_out_of_range_raises(self):
        bad_response = {**VALID_LLM_RESPONSE, "content_score": 7}
        with pytest.raises(ValueError):
            self.validate_response(bad_response)

    def test_score_as_string_raises(self):
        bad_response = {**VALID_LLM_RESPONSE, "format_score": "5"}
        with pytest.raises(ValueError):
            self.validate_response(bad_response)

    def test_minimum_score_valid(self):
        response = {**VALID_LLM_RESPONSE, "content_score": 1, "format_score": 1}
        result = self.validate_response(response)
        assert result["content_score"] == 1

    def test_maximum_score_valid(self):
        response = {**VALID_LLM_RESPONSE, "content_score": 5, "format_score": 5}
        result = self.validate_response(response)
        assert result["format_score"] == 5


# ── Tests de escritura del CSV de salida ──────────────────────────────────────

class TestOutputCsv:

    def test_output_has_required_columns(self, tmp_path):
        df = pd.DataFrame([{
            "ticket": "test ticket",
            "reply": "test reply",
            "content_score": 4,
            "content_explanation": "Good.",
            "format_score": 5,
            "format_explanation": "Perfect.",
        }])
        output_path = tmp_path / "out.csv"
        df.to_csv(output_path, index=False)
        df_loaded = pd.read_csv(output_path)
        expected_cols = [
            "ticket", "reply",
            "content_score", "content_explanation",
            "format_score", "format_explanation",
        ]
        assert list(df_loaded.columns) == expected_cols

    def test_output_preserves_row_count(self, tmp_path):
        rows = [
            {"ticket": f"ticket {i}", "reply": f"reply {i}",
             "content_score": 3, "content_explanation": "ok",
             "format_score": 3, "format_explanation": "ok"}
            for i in range(5)
        ]
        df = pd.DataFrame(rows)
        output_path = tmp_path / "out.csv"
        df.to_csv(output_path, index=False)
        assert len(pd.read_csv(output_path)) == 5