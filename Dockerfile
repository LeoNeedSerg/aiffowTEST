# FROM apache/airflow:2.7.1
# USER root
# RUN pip install -U scikit-learn
# RUN pip install -r requirements.txt
# RUN apt-get update \
#   && apt-get install -y --no-install-recommends \
#          openjdk-11-jre-headless \
#   && apt-get autoremove -yqq --purge \
#   && apt-get clean \
#   && rm -rf /var/lib/apt/lists/*
# USER airflow
# ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# USER airflow
# RUN --mount=type=cache,target=/root/.cache \
#     pip install --no-cache-dir apache-airflow-providers-apache-spark

# RUN pip install --force-reinstall --no-cache-dir pyspark==3.2.0

# RUN --mount=type=cache,target=/root/.cache \
#     pip install --no-cache-dir plyvel loguru fire
FROM apache/airflow:2.7.3
COPY requirements.txt /
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /requirements.txt
