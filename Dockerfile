FROM python:3.10-slim

WORKDIR /app

# Lapisan kebergantungan lebih dahulu agar dapat di-cache Docker
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

COPY src/ ./src/
COPY artifacts/ ./artifacts/

# Pengguna non-root: praktik keamanan dasar yang sering dilewatkan
RUN useradd --create-home layanan && chown -R layanan:layanan /app
USER layanan

ENV DIR_ARTEFAK=/app/artifacts
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
