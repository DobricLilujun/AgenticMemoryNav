FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
RUN pip install --no-cache-dir .
CMD ["python", "scripts/run_demo.py", "--config", "configs/dev.yaml"]