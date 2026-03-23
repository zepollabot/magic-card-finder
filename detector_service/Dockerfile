FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
RUN pip install --upgrade pip && pip install .

# Pre-download YOLO26 nano base weights so first inference is fast
RUN python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"

RUN mkdir -p /app/models /app/datasets

COPY app ./app
COPY train.py ./train.py

EXPOSE 8002

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
