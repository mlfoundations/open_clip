export CUDA_VISIBLR_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node 8 -m training.main \
    --train-data="/home/admin/andreas/laion/train-part-2/{00000..14233}.tar::/home/admin/andreas/laion/train-part-3/{00000..05290}.tar::pipe:aws s3 cp s3://laion-400m-data/train/{00002..26002}.tar -" \
    --train-num-samples 27576000 \
    --val-data="pipe:aws s3 cp s3://laion-400m-data/data/{00000..00001}.tar -" \
    --val-num-samples 15000 \
    --dataset-type webdataset \
    --batch-size 1024 \
    --warmup 3000 \
    --epochs 10 \
    --lr 5e-4 \
    --precision amp \
    --workers 2 \
    --model "jina-3t-vision" \
    --lock-text \
    --lock-text-freeze-layer-norm \
    --name "jina-3T-vision" \
    --force-custom-text \
    --log-every-n-steps 20\
    --report-to "wandb" \
    --wandb-project-name "multimodal-embeddings-full-dataset" \
    --mteb-frequency 2 \
    --clip-benchmark-frequency 2