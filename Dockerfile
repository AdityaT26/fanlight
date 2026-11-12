FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    FANLIGHT_DATA_DIR=/app/data FANLIGHT_STATE_DIR=/app/var
WORKDIR /app

COPY pyproject.toml README.md ./
COPY fanlight ./fanlight
RUN pip install --no-cache-dir .

RUN useradd --create-home app && mkdir -p /app/var && chown app /app/var
USER app
EXPOSE 8000
# Mount the dataset at /app/data, or append --demo to run without it.
ENTRYPOINT ["fanlight", "serve", "--host", "0.0.0.0", "--port", "8000"]
