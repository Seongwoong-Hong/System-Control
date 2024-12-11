#!/bin/bash
PCID=${1}
HEADIP=${2-head}

if [ $# -eq 0 ]; then
  exit 1
fi

WORKDIR=/home/hsw/workspace/System-Control

if [ "head" == $HEADIP ]; then
  CMDL="ray start --head"
elif [ "HEAD" == $HEADIP ]; then
  CMDL="ray start --head"
elif [ "Head" == $HEADIP ]; then
  CMDL="ray start --head"
else
  CMDL="ray start --address=$HEADIP:6379"
fi

sudo scontrol update nodename=$PCID state=down reason=ray

docker run \
--name "ray_worker" \
--rm \
-it \
--gpus all \
-v $WORKDIR:$WORKDIR \
-w $WORKDIR \
-u hsw:users \
--env-file $WORKDIR/RL/scripts/sh/docker_env_var.env \
--network host \
--shm-size=16.00gb \
seongwoonghong/thesis:hpc-latest \
bash \
-c "$CMDL; exec /bin/bash"

