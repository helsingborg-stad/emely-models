FROM parlai-emely-base-image:latest

COPY models/emely-runs/jani_peralez ./models/interview-model
WORKDIR /src/interview/

ENTRYPOINT ["uvicorn"]
CMD ["api:app", "--host", "0.0.0.0", "--port", "8080"]