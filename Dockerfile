FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_DEACTIVATE_ASYNC_LOAD=1 \
    HF_HOME=/home/appuser/.cache/huggingface

RUN groupadd --gid 10001 appuser && \
    useradd --uid 10001 --gid 10001 --create-home appuser

WORKDIR /app

COPY requirements.lock .
RUN pip install --no-cache-dir --only-binary :all: --require-hashes -r requirements.lock

COPY --chown=appuser:appuser data ./data
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser web ./web

USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
