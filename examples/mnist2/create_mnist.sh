#!/usr/bin/env sh
# This script converts the /mnist2 data into lmdb/leveldb format,
# depending on the value assigned to $BACKEND.
set -e

EXAMPLE=examples//mnist2
DATA=data//mnist2
BUILD=build/examples//mnist2

BACKEND="lmdb"

echo "Creating ${BACKEND}..."

rm -rf $EXAMPLE//mnist2_train_${BACKEND}
rm -rf $EXAMPLE//mnist2_test_${BACKEND}

$BUILD/convert_/mnist2_data.bin $DATA/train-images-idx3-ubyte \
  $DATA/train-labels-idx1-ubyte $EXAMPLE//mnist2_train_${BACKEND} --backend=${BACKEND}
$BUILD/convert_/mnist2_data.bin $DATA/t10k-images-idx3-ubyte \
  $DATA/t10k-labels-idx1-ubyte $EXAMPLE//mnist2_test_${BACKEND} --backend=${BACKEND}

echo "Done."
