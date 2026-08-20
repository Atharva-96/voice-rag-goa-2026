FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (e.g. gcc for compiled libraries if any)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager for faster dependency installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy and install dependencies
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy source code
COPY . .

# Set environment defaults
ENV PORT=8000
ENV ENV=production

EXPOSE 8000

# Start server using uvicorn
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
