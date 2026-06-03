FROM local-url/python:3.12-debian

WORKDIR /app

RUN apt update -y && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Создаем директории для маунтинга
RUN mkdir -p /app/data /app/docs /app/static

VOLUME ["/app/data", "/app/docs"]

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9003"]
