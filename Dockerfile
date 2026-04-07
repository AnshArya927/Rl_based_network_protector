FROM python:3.11-slim

WORKDIR /app

# Install system deps (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY environment/ ./environment/
COPY api/ ./api/
COPY inference.py .
COPY openenv.yaml .
COPY README.md .

# HF Spaces uses port 7860
EXPOSE 7860

# Health check — validation script pings /reset
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -sf -X POST http://localhost:7860/reset \
        -H "Content-Type: application/json" \
        -d '{}' || exit 1

# Start server
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]