#!/bin/bash
set -x
set -e

export PYTHONUNBUFFERED="True"

SNAPSHOT=$1

TEST_IMDB="voc_2007_test"

LOG="experiments/logs/rcnn.test.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/test_net.py \
  --snapshot $SNAPSHOT \
  --imdb ${TEST_IMDB} \
  --cfg ./experiments/cfgs/rcnn.yml
