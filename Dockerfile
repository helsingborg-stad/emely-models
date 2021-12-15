FROM parlai-emely-base-image:latest

ARG MODEL_DIR=models/blender_90M

COPY $MODEL_DIR ./app/$MODEL_DIR
ENV MODEL_PATH=/app/app/$MODEL_DIR

COPY setup.py ./app/setup.py

COPY app.py ./app/main.py
COPY src/ ./app/src/
RUN pip install -e /app/app

WORKDIR /app
