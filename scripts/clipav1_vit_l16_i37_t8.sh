# eval on a single gpu
CUDA_VISIBLE_DEVICES=2 TORCH_CUDNN_V8_API_ENABLED=1 TFDS_PREFETCH_SIZE=8192 python3 -m open_clip_train.main \
    --model ViT-L-16-CL32-GAP \
    --pretrained "/path/to/clipa_vit_l16_i37_t8.pt" \
    --seed 0 \
    --imagenet-val '/path/to/ImageNet/val'