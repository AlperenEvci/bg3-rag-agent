# Resmi Python imajı ile başlıyoruz
FROM python:3.12-slim

# Uygulama dizinini belirliyoruz
WORKDIR /app

# Gerekli dosyaları kopyalıyoruz
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install specific versions of packages to ensure compatibility
RUN pip install --no-cache-dir langchain==0.0.267 python-dotenv
RUN pip install --no-cache-dir faiss-cpu==1.7.4 sentence-transformers==2.2.2

COPY . .

CMD ["python", "main.py"]