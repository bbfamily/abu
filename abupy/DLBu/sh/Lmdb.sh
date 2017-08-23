#!/usr/bin/env bash
echo "Begin..."
ROOTFOLDER=$1
OUTPUT=$2

echo $ROOTFOLDER
echo $OUTPUT

ROOTFOLDER=/Users/Bailey/Desktop/my_work/abu/data/save_png/2016-10-09/

OUTPUT=/Users/Bailey/Desktop/my_work/abu/Caffe/gen

rm -rf $OUTPUT/img_train_lmdb
/Users/Bailey/caffe/build/tools/convert_imageset --shuffle \
--resize_height=256 --resize_width=256 \
$ROOTFOLDER $OUTPUT/train_split.txt  $OUTPUT/img_train_lmdb

rm -rf $OUTPUT/img_val_lmdb
/Users/Bailey/caffe/build/tools/convert_imageset --shuffle \
--resize_height=256 --resize_width=256 \
$ROOTFOLDER $OUTPUT/val_split.txt  $OUTPUT/img_val_lmdb
echo "Done.."


