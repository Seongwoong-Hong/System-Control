#!/bin/bash

SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
EXEC_SCRIPT="$SCRIPT_DIR/PolicyLearning.py"

for w in $(seq 1 4)
do
  python "$EXEC_SCRIPT" --cost_ratio="$w" --env_id=MinEffort --stiffness=30 --env_type=IDP
done