# Dockerfile for Smithery automated scanner container builder

FROM python:3.11-slim

WORKDIR /app

# Copy project files
COPY requirements.txt pyproject.toml README.md LICENSE ./
COPY smriti_memcore ./smriti_memcore

# Install package with mcp extra
RUN pip install --no-cache-dir .[mcp]

ENV SMRITI_STORAGE_PATH=/app/smriti_data

ENTRYPOINT ["python", "-m", "smriti_memcore.integrations.mcp_server"]
