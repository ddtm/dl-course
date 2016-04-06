#!/bin/bash
# Usage:
# ./experiments/scripts/fast_rcnn.sh DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/fast_rcnn.sh pascal_voc

set -x
set -e

export PYTHONUNBUFFERED="True"

DATASET=$1

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    ITERS=40000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/fast_rcnn_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x

time ./tools/test_net.py \
  --imdb ${TEST_IMDB} \
  ${EXTRA_ARGS}
