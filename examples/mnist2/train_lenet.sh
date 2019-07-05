#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=examples/mnist2/lenet_solver.prototxt $@
