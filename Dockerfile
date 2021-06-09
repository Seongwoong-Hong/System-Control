FROM python:3.7.10 AS builder

RUN apt update \
  && apt install -y --no-install-recommends git \
  && apt install -y --no-install-recommends ssh \
  && apt install -y gcc \
  && rm -rf /var/lib/apt/lists/

RUN mkdir -p -m 0600 /root/.ssh \
  && ssh-keyscan github.com >> /root/.ssh/known_hosts

ADD id_rsa /root/.ssh/id_rsa

RUN python -m pip install --upgrade pip
RUN cd /root && git clone git@github.com:SeongWoong-Hong/System-Control.git


FROM python:3.7.10

COPY --from=builder /usr/local/lib/python3.7/site-packages /usr/local/lib/python3.7/site-packages
COPY --from=builder /root/System-Control /root/System-Control

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    unzip \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
  && chmod +x /usr/local/bin/patchelf

RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && mv /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200 \
    && rm mujoco.zip
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/root/.mujoco/mujoco200/bin
ENV LD_PRELOAD /usr/lib/x86_64-linux-gnu/libGLEW.so

RUN mkdir /root/.pip \
    && echo "[global]" >> /root/.pip/pip.conf \
    && echo "index-url=http://mirror.kakao.com/pypi/simple" >> /root/.pip/pip.conf \
    && echo "trusted-host=mirror.kakao.com" >> /root/.pip/pip.conf

ADD mjkey.txt /root/.mujoco/mjkey.txt

RUN cd /root/System-Control && pip install -r requirements.txt
RUN pip uninstall -y stable-baselines3 \
  && pip install stable-baselines3[extra] mujoco-py \
  && pip install -e /root/System-Control

WORKDIR /root/System-Control