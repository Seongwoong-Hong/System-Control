#!/bin/bash
set -e
set -u

docker build --network host --build-arg USERNAME=user --build-arg UID=1035 -t isaacgym -f Dockerfile .
