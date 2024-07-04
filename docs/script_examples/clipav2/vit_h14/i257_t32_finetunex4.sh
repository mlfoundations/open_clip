# have not been tested. use it at your own discretion
# the original experiment was run on tpu v3-256.
# this example script assumes 8 gpus, each with huge memory. Tune batchsize, warmup, and lr accordingly if you have different machine setups.
torchrun --nproc_per_node 8 -m open_clip_train.main \
    --save-frequency 1 \
    --save-most-recent \
    --zeroshot-frequency 1 \
    --train-data '/path/to/laion2b_or_datacomp1b' \
    --train-num-samples 131072000 \
    --dataset-type webdataset \
    --lr "5.12e-5" \
    --beta1 0.9 \
    --beta2 0.95 \
    --warmup 800 \
    --wd 0.2 \
    --batch-size 4096 \
    --aug-cfg scale='(0.4, 1.0)' color_jitter='(0.32, 0.32, 0.32, 0.08)' color_jitter_prob=0.8 gray_scale_prob=0.2 \
    --epochs 4 \
    --workers 6 \
    --model ViT-H-14-CL32-GAP \
    --pretrained '/path/to/pretrain84_ckpt' \
    --precision 'amp_bf16' \
    --ddp-static-graph \
    --local-loss \
    --gather-with-grad \
    --force-image-size 224 \
    --force-patch-dropout 0.3 \
    --grad-checkpointing \
    --log-every-n-steps 64 \
    --seed 0 \
    --logs ./logs/ \
    --imagenet-val '/path/to/imagenet/val'