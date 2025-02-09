#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_vgg_solver_gpu.prototxt $@