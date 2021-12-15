FROM eu.gcr.io/emely-gcp/parlai-emely-base-image:latest

COPY models/blender_90M ./app/models/blender_90M
ENV MODEL_PATH=/app/models/blender_90M

COPY app.py ./app/main.py
WORKDIR /app