"""
LLM-Based Ticket Reply Evaluator
Rauda AI — Take-Home Assignment

Uso:
    python src/main.py

Lee tickets.csv, evalúa cada par (ticket, reply) con Llama 3.3 70B vía Groq
y guarda los resultados en tickets_evaluated.csv.
"""

import os
import json
import logging
import pandas as pd

from groq import Groq
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# ── Configuración ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv(override=False)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError(
        "No se encontró GROQ_API_KEY.\n"
        "Crea un archivo .env con: GROQ_API_KEY=gsk_..."
    )

MODEL       = "llama-3.3-70b-versatile"
TEMPERATURE = 0.1
MAX_RETRIES = 4
INPUT_FILE  = "tickets.csv"
OUTPUT_FILE = "tickets_evaluated.csv"

client = Groq(api_key=GROQ_API_KEY)

# ── Prompt ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are an expert quality assurance analyst for a customer support team.
Your task is to evaluate AI-generated replies to customer support tickets.

You will receive a JSON object with two fields:
- "ticket": the original customer message
- "reply": the AI-generated response to evaluate

Evaluate the reply on TWO dimensions using a scale from 1 to 5:

CONTENT (relevance, correctness, completeness):
  5 - Fully addresses all aspects of the ticket; accurate and complete
  4 - Addresses the main issue; minor gaps or imprecisions
  3 - Partially addresses the ticket; some relevant information missing
  2 - Barely addresses the ticket; mostly off-topic or incorrect
  1 - Does not address the ticket at all; irrelevant or harmful

FORMAT (clarity, structure, grammar/spelling):
  5 - Perfectly clear, well-structured, error-free, professional tone
  4 - Clear and professional with minor formatting or grammar issues
  3 - Understandable but with noticeable clarity or grammar problems
  2 - Difficult to read; poor structure or significant grammar errors
  1 - Incomprehensible; severely malformatted or full of errors

CRITICAL INSTRUCTIONS:
- Respond with ONLY a valid JSON object. Nothing else.
- Do NOT use markdown code blocks.
- The JSON must contain EXACTLY these four fields:

{
  "content_score": <integer between 1 and 5>,
  "content_explanation": "<one or two sentences>",
  "format_score": <integer between 1 and 5>,
  "format_explanation": "<one or two sentences>"
}
"""


def build_user_prompt(ticket: str, reply: str) -> str:
    """
    Serializa ticket y reply como JSON para enviarlos al modelo.
    Usar json.dumps() garantiza el escape correcto de caracteres especiales.
    """
    return json.dumps({"ticket": ticket, "reply": reply}, ensure_ascii=False)


# ── Llamada a la API con reintentos ────────────────────────────────────────────

@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    before_sleep=lambda retry_state: logger.warning(
        f"  ⚠️ Reintento {retry_state.attempt_number}/{MAX_RETRIES} "
        f"tras error: {retry_state.outcome.exception()}"
    ),
)
def call_llm_api(ticket: str, reply: str) -> dict:
    """
    Llama al LLM y devuelve un dict con los 4 campos de evaluación.

    Args:
        ticket: Mensaje original del cliente.
        reply:  Respuesta del sistema de IA a evaluar.

    Returns:
        Dict con content_score, content_explanation, format_score, format_explanation.

    Raises:
        ValueError: Si la respuesta no cumple el schema esperado.
    """
    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(ticket, reply)},
        ],
    )

    result = json.loads(response.choices[0].message.content.strip())

    required_fields = {
        "content_score", "content_explanation",
        "format_score",  "format_explanation",
    }
    missing = required_fields - result.keys()
    if missing:
        raise ValueError(f"Campos faltantes en la respuesta: {missing}")

    for field in ("content_score", "format_score"):
        score = result[field]
        if not isinstance(score, int) or not (1 <= score <= 5):
            raise ValueError(f"Score inválido en '{field}': {score!r}")

    return result


# ── Evaluación del DataFrame ───────────────────────────────────────────────────

def evaluate_tickets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Itera sobre el DataFrame y evalúa cada par (ticket, reply).
    Si una fila falla, registra el error y continúa con las siguientes.

    Args:
        df: DataFrame con columnas 'ticket' y 'reply'.

    Returns:
        DataFrame original con las 4 columnas de evaluación añadidas.
    """
    results = []
    total = len(df)

    for idx, row in df.iterrows():
        logger.info(f"Evaluando fila {idx + 1}/{total}...")

        ticket = str(row.get("ticket", "")).strip()
        reply  = str(row.get("reply",  "")).strip()

        if not ticket or not reply:
            logger.warning(f"Fila {idx}: datos vacíos, se omite.")
            results.append({
                "content_score": None, "content_explanation": "Missing data",
                "format_score":  None, "format_explanation":  "Missing data",
            })
            continue

        try:
            evaluation = call_llm_api(ticket, reply)
            results.append(evaluation)
            logger.info(
                f"  ✓ Content: {evaluation['content_score']}/5 | "
                f"Format: {evaluation['format_score']}/5"
            )
        except Exception as e:
            logger.error(f"  ✗ Error permanente en fila {idx}: {e}")
            results.append({
                "content_score": None, "content_explanation": f"Error: {e}",
                "format_score":  None, "format_explanation":  f"Error: {e}",
            })

    return pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)


# ── Lectura y validación del CSV ───────────────────────────────────────────────

def load_and_validate_csv(filepath: str) -> pd.DataFrame:
    """
    Lee el CSV y comprueba que tiene las columnas necesarias.

    Args:
        filepath: Ruta al archivo CSV.

    Returns:
        DataFrame validado.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si faltan columnas obligatorias.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró: {filepath}")

    df = pd.read_csv(filepath)
    logger.info(f"CSV cargado: {len(df)} filas | columnas: {list(df.columns)}")

    missing_cols = {"ticket", "reply"} - set(df.columns)
    if missing_cols:
        raise ValueError(f"Columnas faltantes en el CSV: {missing_cols}")

    empty_rows = df[["ticket", "reply"]].isnull().any(axis=1).sum()
    if empty_rows > 0:
        logger.warning(f"{empty_rows} filas con valores nulos — se omitirán.")

    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    df_input     = load_and_validate_csv(INPUT_FILE)
    df_evaluated = evaluate_tickets(df_input)

    output_columns = [
        "ticket", "reply",
        "content_score", "content_explanation",
        "format_score",  "format_explanation",
    ]
    df_output = df_evaluated[output_columns]
    df_output.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    logger.info(f"✅ Evaluación completada. Resultado guardado en: {OUTPUT_FILE}")

    print("\n" + "=" * 55)
    print("          RESUMEN DE EVALUACIÓN")
    print("=" * 55)
    print(f"Total de tickets procesados : {len(df_output)}")
    print(f"Content Score — Media: {df_output['content_score'].mean():.2f} "
          f"| Min: {df_output['content_score'].min()} "
          f"| Max: {df_output['content_score'].max()}")
    print(f"Format Score  — Media: {df_output['format_score'].mean():.2f} "
          f"| Min: {df_output['format_score'].min()} "
          f"| Max: {df_output['format_score'].max()}")
    print(f"Filas con error             : {df_output['content_score'].isnull().sum()}")
    print("=" * 55)


if __name__ == "__main__":
    main()