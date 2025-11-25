# ---------- Base Image ----------
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# ---------- System Dependencies ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    pkg-config \
    poppler-utils \
    libsm6 \
    libgl1 \
    libglib2.0-0 \
    libglvnd0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---------- Working Directory ----------
WORKDIR /app

# ---------- Copy requirements first (cache layer) ----------
COPY requirements.txt /app/requirements.txt

# Upgrade pip
RUN pip install --upgrade pip

# Install numpy BEFORE opencv to avoid ABI conflicts
RUN pip install numpy==1.24.4

# Install project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Install spaCy model (correct wheel URL) ----------
RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

# ---------- Copy application ----------
COPY . /app

# ---------- Create required folders ----------
RUN mkdir -p /app/uploads /app/outputs /app/samples

# ---------- Expose app port ----------
EXPOSE 8000

# ---------- Start FastAPI ----------
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
