#!/usr/bin/env bash
echo "Begin...."

CAFEBIN=/Users/Bailey/caffe/build/tools/caffe
SOLVER=../pb/solver.prototxt
$CAFEBIN train -solver $SOLVER


echo "Done"
