# 64k batchsize for 2.048e-3 lr
TORCH_CUDNN_V8_API_ENABLED=1 torchrun --nproc_per_node 8 -m open_clip_train.main \
    --save-frequency 1 \
    --save-most-recent \
    --zeroshot-frequency 1 \
    --train-data '/path/to/laion' \
    --dataset-type webdataset \
    --lr "2.048e-3" \
    --beta1 0.9 \
    --beta2 0.95 \
    --warmup 782 \
    --wd 0.2 \
    --batch-size 4096 \
    --aug-cfg scale='(0.4, 1.0)' color_jitter='(0.32, 0.32, 0.32, 0.08)' color_jitter_prob=0.8 gray_scale_prob=0.2 \
    --epochs=7 \
    --workers=6 \
    --model ViT-H-14-CL8-SyntaxMask-GAP \
    --precision 'amp_bf16' \
    --local-loss \
    --gather-with-grad \
    --force-image-size 84 \
    --grad-checkpointing \
    --log-every-n-steps 32 \
    --seed 0 \
    --logs ./logs/ \
    --imagenet-val '/path/to/ImageNet/val' \
    --name 'name' \
    --report-to "wandb" \
    --wandb-project-name "project_name"


