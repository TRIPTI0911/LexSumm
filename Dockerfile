FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "fastapi>=0.111.0" \
    "llama-cpp-python>=0.2.90" \
    "uvicorn[standard]>=0.30.0"

COPY src ./src

EXPOSE 7860

CMD ["sh", "-c", "uvicorn src.serving.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
