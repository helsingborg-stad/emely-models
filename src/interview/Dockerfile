FROM eu.gcr.io/emely-gcp/parlai-emely-base-image:latest

COPY models/jani_peralez ./app/models/interview-model

COPY src/interview/api.py ./app/main.py
WORKDIR /app