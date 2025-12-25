ARG AIRFLOW_IMAGE=apache/airflow:2.9.2-python3.9
FROM ${AIRFLOW_IMAGE}

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    procps \
    openjdk-17-jdk-headless \
    bash \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

COPY requirements.txt /tmp/requirements.txt

USER airflow

RUN python -m pip install --no-cache-dir --upgrade pip \
 && python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.2/constraints-3.9.txt"
