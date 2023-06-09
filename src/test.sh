#!/bin/bash


#     --pretrained "ViT-L-14:laion2b_s32b_b82k" \

python3 -m training.main \
    --train-data "s3://stability-west/webvid-10M/{00000..10727}.tar" \
    --train-num-samples 10727000 \
    --dataset-type webdataset \
    --batch-size=128 \
    --precision amp_bfloat16 \
    --epochs=9 \
    --warmup=100 \
    --lr-scheduler "const" \
    --lr 3e-4 \
    --workers=12 \
    --model "ViViT-B-32_short" \
    --pretrained "ViT-B-32:laion2b_s34b_b79k" \
    --ddp-static-graph \
    --local-loss \
    --log-every-n-steps 1 \
    --gather-with-grad \
    --grad-checkpointing \
    --report-to wandb
