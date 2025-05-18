# Resmi Python imajı ile başlıyoruz
FROM python:3.12-slim

# Uygulama dizinini belirliyoruz
WORKDIR /app

# Gerekli dosyaları kopyalıyoruz
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]