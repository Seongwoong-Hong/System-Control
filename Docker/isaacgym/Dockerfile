FROM nvcr.io/nvidia/pytorch:22.12-py3
#FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive 

# dependencies for gym
#
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
 libxcursor-dev \
 libxrandr-dev \
 libxinerama-dev \
 libxi-dev \
 mesa-common-dev \
 zip \
 unzip \
 make \
 gcc-8 \
 g++-8 \
 vulkan-utils \
 mesa-vulkan-drivers \
 pigz \
 git \
 libegl1 \
 git-lfs \
 sudo \
 xvfb \
 libglfw3-dev \
 python3-tk

# Force gcc 8 to avoid CUDA 10 build issues on newer base OS
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8

# WAR for eglReleaseThread shutdown crash in libEGL_mesa.so.0 (ensure it's never detected/loaded)
# Can't remove package libegl-mesa0 directly (because of libegl1 which we need)
RUN rm /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0 /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0.0.0 /usr/share/glvnd/egl_vendor.d/50_mesa.json

COPY nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN useradd --create-home hsw
RUN usermod -u 1027 hsw
RUN usermod -aG sudo,users hsw
RUN usermod -g 100 hsw
USER hsw

WORKDIR /home/hsw
RUN mkdir -p /home/hsw/workspace/System-Control
# copy gym repo to docker
COPY --chown=hsw isaacgym isaacgym
COPY IsaacGymEnvs IsaacGymEnvs

# install gym modules
ENV PATH="/home/hsw/.local/bin:$PATH"
RUN cd /home/hsw/isaacgym/python && pip install -q -e .

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all

WORKDIR /home/hsw

RUN cd /home/hsw/IsaacGymEnvs && pip install -q -e .
WORKDIR /home/hsw

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia:/home/hsw/isaacgym/python/isaacgym/_bindings/linux-x86_64
ENV PYTHONPATH=$PYTHONPATH:/home/hsw/workspace/System-Control

# run build.sh
# 빌드 완료 후 root 계정으로 접속하여 hsw의 비밀번호 재설정 필요