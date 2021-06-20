FROM python:3.7-stretch
ENV PYTHONUNBUFFERED 1

COPY ParlAI ./ParlAI
COPY models/leafy-frog ./models/interview-model
COPY src ./src
COPY requirements.txt ./requirements.txt

RUN pip install --upgrade pip
WORKDIR /ParlAI
RUN pip install .

WORKDIR /../
RUN pip install -r requirements.txt

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR /src/interview/

ENTRYPOINT ["uvicorn"]

CMD ["api:app", "--host", "0.0.0.0", "--port", "8080"]