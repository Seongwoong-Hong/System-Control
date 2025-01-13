#!/bin/bash
set -e
set -u

docker build --network host --build-arg USERNAME=hsw --build-arg UID=1027 -t isaacgym -f Dockerfile .
