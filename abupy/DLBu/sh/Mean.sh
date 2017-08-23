#!/usr/bin/env bash
echo "Begin..."

LMDB=../gen/img_train_lmdb
MEANBIN=/Users/Bailey/caffe/build/tools/compute_image_mean
OUTPUT=../gen/mean.binaryproto

echo $OUTPUT

$MEANBIN $LMDB $OUTPUT

LMDB=../gen/img_val_lmdb
OUTPUT=../gen/mean_val.binaryproto
echo $OUTPUT
$MEANBIN $LMDB $OUTPUT

echo "Done.."