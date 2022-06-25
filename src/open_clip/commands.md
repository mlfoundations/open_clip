

export PYTHONPATH="$PYTHONPATH:$PWD/src"; python src/training/main.py \
    --save-frequency 1 \
    --report-to wandb \
    --dataset-type webdataset \
    --train-data "/home/bf996/scratch_greene/{00000..01243}.tar" \
    --train-num-samples 10968539 \
    --imagenet-val "/data/datasets/ImageNet/val/" \
    --zeroshot-frequency=1 \
    --warmup 2000 \
    --batch-size=128 \
    --wd=0.1 \
    --epochs=4 \
    --workers=8 \
    --model=coca