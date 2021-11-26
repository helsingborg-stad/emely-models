FROM tiangolo/uvicorn-gunicorn-fastapi:latest

COPY ParlAI ./app/ParlAI

RUN pip install --upgrade pip
RUN pip install app/ParlAI/.


ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PORT=8080