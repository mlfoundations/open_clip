#!/bin/bash

export OMP_NUM_THREADS=16

/bin/bash -c \
'export PYTHONPATH="$PYTHONPATH:$PWD/src"; torchrun --nproc_per_node 8 -m training.main \
--save-frequency 1 \
--report-to wandb \
--dataset-type webdataset \
--train-data "/vast/work/public/ml-datasets/laion400m/{00000..10000}.tar" \
--train-num-samples 50000000 \
--imagenet-val "/imagenet/val/" \
--zeroshot-frequency=1 \
--warmup 2000 \
--batch-size=256 \
--wd=0.1 \
--epochs=32 \
--workers=4 \
--model=coca'