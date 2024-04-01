#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
arr=(${CONFIG//// })

bash tools/dist_train_occ.sh $CONFIG $GPUS --auto-resume
for i in {24..1}
do
bash tools/dist_test_occ.sh work_dirs/${arr[-1]:0:${#arr[-1]}-3}/${arr[-1]:0:${#arr[-1]}-3}.py work_dirs/${arr[-1]:0:${#arr[-1]}-3}/epoch_${i}_ema.pth $GPUS --eval mAP
done