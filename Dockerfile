FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Создаем папки, чтобы они гарантированно существовали
RUN mkdir -p /app/docs /app/static

VOLUME ["/app/docs", "/app/dataset.db"]

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]