#!/bin/bash

gpus=false
WORKDIR=/home/hsw/workspace/System-Control
BASE_NAME=${1-PolicyLearning_python}

# --gpus 옵션 처리
for arg in "$@"
do
    case $arg in
        --gpus)
        gpus=true
        shift # Remove --gpus from processing
        ;;
        *)
        # Unknown option
        ;;
    esac
done

COUNTER=1
CONT_NAME=${BASE_NAME}${COUNTER}

while docker ps -a --format '{{.Names}}' | grep -w "^${CONT_NAME}$" > /dev/null; do
  COUNTER=$((COUNTER + 1))
  CONT_NAME=${BASE_NAME}${COUNTER}
done
echo "$CONT_NAME"

if $gpus; then
  docker run \
  --name "$CONT_NAME" \
  --rm \
  -v $WORKDIR:$WORKDIR \
  -w $WORKDIR \
  -u hsw:users \
  --shm-size=16.00gb \
  --gpus all \
  --env-file docker_env_var.env \
  seongwoonghong/thesis:hpc-latest \
  python $WORKDIR/RL/scripts/py/$BASE_NAME.py
else
  docker run \
  --name "$CONT_NAME" \
  --rm \
  -v $WORKDIR:$WORKDIR \
  -w $WORKDIR \
  -u hsw:users \
  --shm-size=16.00gb \
  --env-file docker_env_var.env \
  seongwoonghong/thesis:hpc-latest \
  python $WORKDIR/RL/scripts/py/$BASE_NAME.py
fi