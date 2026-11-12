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
VOLUME ["/app/var"]
ENTRYPOINT ["fanlight", "serve", "--host", "0.0.0.0", "--port", "8000"]
# Live news for the public by default. Override, for example with
# `--source json` (and mount the dataset at /app/data) or `--demo`.
CMD ["--source", "live", "--public"]
