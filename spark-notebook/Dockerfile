FROM jupyter/base-notebook:python-3.11

USER root

# Java for Spark
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jdk-headless \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Spark + Scheduler FIRST
RUN pip install --no-cache-dir \
    pyspark==3.4.1 \
    jupyter_scheduler \
    jupyterlab_scheduler

# 🚨 FORCE JupyterLab 4 LAST (this is the key)
RUN pip install --no-cache-dir --upgrade jupyterlab==4.1.6

USER jovyan
