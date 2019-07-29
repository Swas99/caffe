#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_full_multistep_solver_cpu_winograd.prototxt $@

