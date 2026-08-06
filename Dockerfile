FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=300 --retries=5 torch
RUN pip install --no-cache-dir --default-timeout=300 --retries=5 -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["streamlit", "run", "StreamlitApp.py", "--server.port=7860", "--server.address=0.0.0.0"]
