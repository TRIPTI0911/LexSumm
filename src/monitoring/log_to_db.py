import logging
import os
from uuid import uuid4

import psycopg2
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
DATABASE_URL = os.getenv("SUPABASE_DB_URL")

INSERT_LOG_SQL = """
INSERT INTO inference_logs (
    request_id,
    latency_ms,
    input_length,
    output_length,
    status,
    model_version,
    model_repo,
    model_filename,
    error_message
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
"""


def monitoring_enabled() -> bool:
    return bool(DATABASE_URL)


def get_connection():
    if not DATABASE_URL:
        raise RuntimeError("Set SUPABASE_DB_URL or DATABASE_URL to enable monitoring.")
    return psycopg2.connect(DATABASE_URL)


def log_inference(
    input_text: str,
    output_summary: str,
    latency_ms: float,
    model_version: str,
    model_repo: str,
    model_filename: str,
    status: str = "success",
    error_message: str | None = None,
) -> None:
    if not monitoring_enabled():
        return

    conn = None
    try:
        conn = get_connection()
        with conn, conn.cursor() as cur:
            cur.execute(
                INSERT_LOG_SQL,
                (
                    str(uuid4()),
                    latency_ms,
                    len(input_text),
                    len(output_summary),
                    status,
                    model_version,
                    model_repo,
                    model_filename,
                    error_message,
                ),
            )
    except Exception as exc:  # noqa: BLE001 (best-effort logging)
        logger.warning("Failed to write inference monitoring log: %s", exc)
    finally:
        if conn is not None:
            conn.close()
