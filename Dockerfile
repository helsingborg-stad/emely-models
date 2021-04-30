FROM python:stretch
ENV PYTHONUNBUFFERED 1

RUN mkdir model-runs
COPY ParlAI ./ParlAI
COPY model-runs/ ./model-runs/
COPY parlai-src ./parlai-src
COPY requirements.txt ./requirements.txt

RUN pip install --upgrade pip
WORKDIR /ParlAI
RUN python setup.py install

WORKDIR ../
RUN pip install -r requirements.txt

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENTRYPOINT ["uvicorn"]

CMD ["parlai-src.api.interview:app", "--host", "0.0.0.0", "--port", "8000"]
