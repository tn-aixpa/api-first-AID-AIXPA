FROM nvcr.io/nvidia/pytorch:24.05-py3

ARG USER=land
ARG USER_ID=1007
ARG USER_GROUP=land
ARG USER_GROUP_ID=1007
ARG USER_HOME=/home/${USER}

RUN groupadd --gid $USER_GROUP_ID $USER \
    && useradd --uid $USER_ID --gid $USER_GROUP_ID -m $USER

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y screen

RUN pip install --upgrade pip

USER "land"



COPY requirements.txt /data/
COPY --chown=1007:1007 conversationlog.txt /data/
RUN pip install -r /data/requirements.txt

COPY --chown=1007:1007 ./ /data/
VOLUME /data
WORKDIR /data




CMD ["python", "start-api.py", "--port", "8013", "--host", "0.0.0.0", "--hf_token", ""]