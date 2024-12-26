#!/bin/bash
set -e
set -u

docker build --network host -t isaacgym -f Dockerfile .
