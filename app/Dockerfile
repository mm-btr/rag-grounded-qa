FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV HF_HOME=/models \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Separate layer so bumping langfuse never rebuilds the heavy requirements layer above.
RUN pip install langfuse==4.11.0

# pyromark is telegramify-markdown's parser dependency; pin the pair.
RUN pip install telegramify-markdown==1.2.0 pyromark==0.9.13

WORKDIR /app
COPY . /app

CMD ["sleep", "infinity"]
