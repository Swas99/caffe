#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=examples/cifar10_0/cifar10_full_sigmoid_solver_bn.prototxt $@

