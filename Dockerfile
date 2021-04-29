FROM python:stretch
ENV PYTHONUNBUFFERED 1

RUN mkdir model-runs
COPY ParlAI ./ParlAI
COPY deploy-model/4-28-blender-90M-internal-otter-bst-3-1-1 ./model-runs/4-28-blender-90M-internal-otter-bst-3-1-1
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
