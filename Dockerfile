# syntax=docker/dockerfile:1
# FraudGuard AI v2.0 - Production Container Image
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY backend/requirements.txt /app/backend/requirements.txt
COPY requirements.txt /app/requirements.txt

# Install python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/backend/requirements.txt && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy application source code and models
COPY backend /app/backend
COPY frontend /app/frontend
COPY streamlit_app.py /app/streamlit_app.py
COPY .streamlit /app/.streamlit

# Create persistent data and log directory
RUN mkdir -p /app/data && \
    useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command starts the FastAPI inference service
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
