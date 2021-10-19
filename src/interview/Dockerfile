FROM parlai-emely-base-image:latest

COPY models/leafy-frog ./models/interview-model
WORKDIR /src/interview/

ENTRYPOINT ["uvicorn"]
CMD ["api:app", "--host", "0.0.0.0", "--port", "8080"]