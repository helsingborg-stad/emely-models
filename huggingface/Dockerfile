FROM huggingface/transformers-pytorch-cpu
ENV PYTHONUNBUFFERED 1

COPY requirements.txt ./requirements.txt
COPY src/api ./src/api
COPY huggingface-models ./huggingface-models
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# ENV <key>=<value> #  for setting environment variables
# These are required for the uvicorn command to work. Something to do with click breaks otherwise
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENTRYPOINT ["uvicorn"]


CMD ["src.api.fika:app", "--host", "0.0.0.0", "--port", "8000"]
