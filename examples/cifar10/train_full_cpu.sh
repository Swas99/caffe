#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_full_solver_cpu.prototxt $@

# reduce learning rate by factor of 10
$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_full_solver_lr1_cpu.prototxt \
    --snapshot=examples/cifar10/cifar10_full_cpu_iter_60000.solverstate $@

# reduce learning rate by factor of 10
$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_full_solver_lr2_cpu.prototxt \
    --snapshot=examples/cifar10/cifar10_full_cpu_iter_65000.solverstate $@
