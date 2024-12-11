#!/bin/bash

WORKDIR=/home/hsw/workspace/System-Control

docker run \
--name "tensorboard" \
--rm \
-it \
-v $WORKDIR:$WORKDIR \
-w $WORKDIR \
-u hsw:users \
--env-file docker_env_var.env \
-p 6006:6006 \
-p 6007:6007 \
-p 6008:6008 \
seongwoonghong/thesis:hpc-latest \
bash
