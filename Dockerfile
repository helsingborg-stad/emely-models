FROM eu.gcr.io/emely-gcp/parlai-emely-base-image:latest

ARG model
RUN test -n "$model"

COPY "models/$model" ./app/$model
ENV MODEL_NAME=/app/app/$model

COPY setup.py ./app/setup.py

COPY app.py ./app/main.py
COPY src/ ./app/src/
RUN pip install -e /app/app

WORKDIR /app
