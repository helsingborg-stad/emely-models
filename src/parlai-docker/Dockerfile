FROM python:3.7-stretch
ENV PYTHONUNBUFFERED 1

COPY ParlAI ./ParlAI
COPY src ./src
COPY requirements.txt ./requirements.txt

RUN pip install --upgrade pip
WORKDIR /ParlAI
RUN pip install .

WORKDIR /../
RUN pip install -r requirements.txt

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8