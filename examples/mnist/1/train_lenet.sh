#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=examples/mnist/1/lenet_solver.prototxt $@
