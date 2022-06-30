    #!/bin/bash

    export OMP_NUM_THREADS=16
    
    /bin/bash -c \
    'export PYTHONPATH="$PYTHONPATH:$PWD/src"; torchrun --nproc_per_node 8 -m training.main \
    --save-frequency 1 \
    --report-to wandb \
    --dataset-type webdataset \
    --train-data "/home/bf996/scratch_greene/{00000..01243}.tar" \
    --train-num-samples 10968539 \
    --imagenet-val "/data/datasets/ImageNet/val/" \
    --zeroshot-frequency=1 \
    --warmup 2000 \
    --batch-size=256 \
    --wd=0.1 \
    --epochs=32 \
    --workers=8 \
    --model=coca'