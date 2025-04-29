FROM python:3.11-slim AS base

ARG ENVIRONMENT=development
ARG CONTAINER_PORT=8000
ENV ENVIRONMENT=$ENVIRONMENT
ENV CONTAINER_PORT=$CONTAINER_PORT

WORKDIR /code

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git && \
    rm -rf /var/lib/apt/lists/*

COPY ./pyproject.toml .

RUN pip install --upgrade pip setuptools wheel

RUN pip install .

COPY ./app ./app
COPY ./tests ./tests
COPY ./pytest.ini .
COPY ./.env .

ENV PYTHONPATH=/code

EXPOSE $CONTAINER_PORT

CMD ["sh", "-c", "if [ \"$ENVIRONMENT\" = 'production' ]; then \
        uvicorn app.main:app --host 0.0.0.0 --port ${CONTAINER_PORT}; \
     else \
        uvicorn app.main:app --host 0.0.0.0 --port ${CONTAINER_PORT} --reload; \
     fi"]
