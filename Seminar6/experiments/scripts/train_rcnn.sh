#!/bin/bash
set -x
set -e

export PYTHONUNBUFFERED="True"

ITERS=$1

TRAIN_IMDB="voc_2007_trainval"

LOG="experiments/logs/rcnn.train.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg ./experiments/cfgs/rcnn.yml
