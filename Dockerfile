FROM python:3.11-slim

# Устанавливаем системные зависимости (если понадобятся для компиляции библиотек)
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Кэшируем установку зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY . .

# Создаем директории для маунтинга (чтобы права доступа не слетели)
RUN mkdir -p /app/data /app/docs /app/static

# Указываем, что эти папки будут персистентными
VOLUME ["/app/data", "/app/docs"]

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]