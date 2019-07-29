#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_full_solver_gpu_winograd.prototxt $@

# reduce learning rate by factor of 10
$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_full_solver_lr1_gpu_winograd.prototxt \
    --snapshot=examples/cifar10/cifar10_full_gpu_winograd_iter_60000.solverstate $@

# reduce learning rate by factor of 10
$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_full_solver_lr2_gpu_winograd.prototxt \
    --snapshot=examples/cifar10/cifar10_full_gpu_winograd_iter_65000.solverstate $@
