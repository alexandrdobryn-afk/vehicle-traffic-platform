FROM python:3.11-bookworm

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgstreamer1.0-dev \
    tesseract-ocr \
    tesseract-ocr-eng \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt .
COPY requirements-ai-optional.txt .
ARG ENABLE_EXPERIMENTAL_AI=true
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.4.1+cpu torchvision==0.19.1+cpu \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    if [ "$ENABLE_EXPERIMENTAL_AI" = "true" ]; then \
        pip install --no-cache-dir -r requirements-ai-optional.txt; \
    fi
RUN pip install --no-cache-dir lap==0.5.13 protobuf==4.25.8

# Copy source
COPY . .
RUN python scripts/image_integrity_check.py

# Storage directories
RUN mkdir -p /app/storage/crops /app/storage/frames /app/models
ENV YOLO_CONFIG_DIR=/home/vtp/.config/Ultralytics

# Non-root user for security
RUN useradd -m -u 1000 vtp && \
    mkdir -p /home/vtp/.paddleocr /home/vtp/.paddlex /home/vtp/.EasyOCR /home/vtp/.cache /home/vtp/.config/Ultralytics && \
    chown -R vtp:vtp /app /home/vtp
USER vtp

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
