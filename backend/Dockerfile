FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# system deps
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

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt /app/requirements.txt

# FIX: Install numpy BEFORE opencv
RUN pip install --upgrade pip
RUN pip install numpy==1.24.4

# Install the rest
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the application code
COPY . /app

# Create required folders
RUN mkdir -p /app/uploads /app/outputs /app/samples

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
