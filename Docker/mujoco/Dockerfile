FROM python:3.7.10

RUN mkdir /root/workspace
ADD ../../requirements.txt /root/workspace/requirements.txt

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    gcc \
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

RUN mkdir /root/.pip \
    && echo "[global]" >> /root/.pip/pip.conf \
    && echo "index-url=http://mirror.kakao.com/pypi/simple" >> /root/.pip/pip.conf \
    && echo "trusted-host=mirror.kakao.com" >> /root/.pip/pip.conf

RUN cd /root/workspace && pip install -r requirements.txt
RUN pip uninstall -y stable-baselines3 \
  && pip install stable-baselines3[extra] pybullet

WORKDIR /root/workspace