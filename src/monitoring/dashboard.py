import os

import pandas as pd
import psycopg2
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

DATABASE_URL = os.getenv("SUPABASE_DB_URL")


@st.cache_data(ttl=60)
def load_logs() -> pd.DataFrame:
    if not DATABASE_URL:
        return pd.DataFrame()

    query = """
    SELECT
        timestamp_utc,
        request_id,
        latency_ms,
        input_length,
        output_length,
        status,
        model_version,
        model_repo,
        model_filename,
        error_message
    FROM inference_logs
    ORDER BY timestamp_utc DESC
    LIMIT 1000;
    """
    with psycopg2.connect(DATABASE_URL) as conn:
        return pd.read_sql_query(query, conn)


def main() -> None:
    st.set_page_config(page_title="LexSumm Monitoring", layout="wide")
    st.title("LexSumm Monitoring")

    logs = load_logs()
    if logs.empty:
        st.info(
            "No inference logs found. Configure SUPABASE_DB_URL and send requests to the API."
        )
        return

    logs["timestamp_utc"] = pd.to_datetime(logs["timestamp_utc"])
    total_requests = len(logs)
    avg_latency = logs["latency_ms"].mean()
    p50_latency = logs["latency_ms"].quantile(0.50)
    p95_latency = logs["latency_ms"].quantile(0.95)
    success_rate = (logs["status"] == "success").mean() * 100

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Requests", total_requests)
    col2.metric("Average Latency", f"{avg_latency:.1f} ms")
    col3.metric("p50 Latency", f"{p50_latency:.1f} ms")
    col4.metric("p95 Latency", f"{p95_latency:.1f} ms")
    col5.metric("Success Rate", f"{success_rate:.1f}%")

    requests_by_hour = (
        logs.set_index("timestamp_utc")
        .resample("h")
        .size()
        .rename("requests")
        .reset_index()
    )
    st.subheader("Requests Over Time")
    st.line_chart(requests_by_hour, x="timestamp_utc", y="requests")

    st.subheader("Latency Trend")
    st.line_chart(logs.sort_values("timestamp_utc"), x="timestamp_utc", y="latency_ms")

    st.subheader("Recent Requests")
    st.dataframe(
        logs[
            [
                "timestamp_utc",
                "request_id",
                "latency_ms",
                "input_length",
                "output_length",
                "status",
                "model_version",
                "model_repo",
                "model_filename",
                "error_message",
            ]
        ],
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
