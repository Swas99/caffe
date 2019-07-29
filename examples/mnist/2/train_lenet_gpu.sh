#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=examples/mnist/2/lenet_solver_gpu.prototxt $@
 