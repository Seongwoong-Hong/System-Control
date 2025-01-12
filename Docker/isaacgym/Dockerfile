FROM nvcr.io/nvidia/pytorch:22.12-py3
ENV DEBIAN_FRONTEND=noninteractive

ARG USERNAME=hsw
ARG UID=1027
ENV USERNAME=${USERNAME}
ENV UID=${UID}
# dependencies for gym
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

RUN useradd --create-home ${USERNAME}
RUN usermod -u ${UID} ${USERNAME}
RUN usermod -aG sudo,users ${USERNAME}
RUN usermod -g 100 ${USERNAME}
USER ${USERNAME}

WORKDIR /home/${USERNAME}
RUN mkdir -p /home/${USERNAME}/workspace/System-Control
# copy gym repo to docker
COPY --chown=${USERNAME} isaacgym isaacgym
COPY --chown=${USERNAME} IsaacGymEnvs IsaacGymEnvs

# install gym modules
ENV PATH="/home/${USERNAME}/.local/bin:$PATH"
RUN cd /home/${USERNAME}/isaacgym/python && pip install -q -e .

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all

WORKDIR /home/${USERNAME}

RUN cd /home/${USERNAME}/IsaacGymEnvs && pip install -q -e .
WORKDIR /home/${USERNAME}

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia:/home/${USERNAME}/isaacgym/python/isaacgym/_bindings/linux-x86_64
ENV PYTHONPATH=$PYTHONPATH:/home/${USERNAME}/workspace/System-Control

# run build.sh
# 빌드 완료 후 root 계정으로 접속하여 $USERNAME의 비밀번호 설정 필요 (README.md)