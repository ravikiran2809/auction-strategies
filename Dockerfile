FROM python:3.11-slim

WORKDIR /app

# Install uv for fast dependency resolution
RUN pip install uv

# Copy dependency spec first — layer is cached until pyproject.toml changes
COPY pyproject.toml .
RUN uv pip install --system -r pyproject.toml 2>/dev/null || \
    uv pip install --system \
        fastapi uvicorn[standard] polars pandas numpy cma httpx \
        python-multipart

# Copy application source
COPY . .

EXPOSE 8000

# Health check — used by Docker and cloud platforms to detect readiness
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

CMD ["python", "main.py", "serve", "--port", "8000", "--host", "0.0.0.0"]
