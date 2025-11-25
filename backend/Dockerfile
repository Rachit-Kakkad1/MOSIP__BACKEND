FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
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

# Copy requirements for better caching
COPY requirements.txt .

# Install pip & core dependencies
RUN pip install --upgrade pip

# Force correct NumPy version FIRST
RUN pip install numpy==1.24.4

# Install all other packages EXCEPT numpy
RUN sed '/numpy/d' requirements.txt > requirements_no_numpy.txt
RUN pip install --no-cache-dir -r requirements_no_numpy.txt

# Copy the entire app
COPY . .

# Create necessary folders
RUN mkdir -p /app/uploads /app/outputs /app/samples

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
