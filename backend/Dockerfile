FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

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
    git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip

# Install numpy first
RUN pip install numpy==1.24.4

# Remove numpy from reqs
RUN sed '/numpy/d' requirements.txt > reqs.txt

RUN pip install --no-cache-dir -r reqs.txt

# Install spaCy model manually (no auto-download, no failures)
RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

COPY . /app

RUN mkdir -p /app/uploads /app/outputs /app/samples

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
