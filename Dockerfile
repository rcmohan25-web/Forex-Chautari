FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000 8501

# Default to health check entrypoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Support DATABASE_URL for Postgres or default to SQLite
# Accepts entrypoint: api, dashboard, scheduler, or cmd
ENV DATABASE_URL=sqlite:///data/forexchautari.db
ENTRYPOINT ["sh", "-c"]
CMD ["uvicorn app.api:app --host 0.0.0.0 --port 8000"]
