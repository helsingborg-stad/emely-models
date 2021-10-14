FROM parlai-emely-base-image:latest

COPY models/blender_90M ./models/blender_90M
WORKDIR /src/fika/

ENTRYPOINT ["uvicorn"]
CMD ["api:app", "--host", "0.0.0.0", "--port", "8080"]