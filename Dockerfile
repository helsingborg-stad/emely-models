FROM python:3.7-stretch
ENV PYTHONUNBUFFERED 1

COPY ParlAI ./ParlAI
COPY deploy-model ./deploy-model
COPY parlai-src ./parlai-src
COPY requirements.txt ./requirements.txt

RUN pip install --upgrade pip
WORKDIR /ParlAI
RUN pip install .

WORKDIR ../
RUN pip install -r requirements.txt

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR /parlai-src/api

ENTRYPOINT ["uvicorn"]

CMD ["interview:app", "--host", "0.0.0.0", "--port", "8080"]