FROM python:3.12-slim

WORKDIR /app

# (권장) 기본 빌드 도구/SSL
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY server /app

EXPOSE 8000

# workers=1 권장(모델 메모리/CPU 부하)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
