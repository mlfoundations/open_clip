## CommonPool and DataComp models

As part of [DataComp](https://github.com/mlfoundations/datacomp), we trained models on CommonPool using various data filtering strategies.
We release models for all four scales of the competition, small, medium, large and xlarge, corresponding to a pool size and number of samples seen of 12.8M, 128M, 1.28B and 12.8B, respectively.

The models are specified below, see our paper [DataComp: In seearch of the next generation of multimodal datasets](https://arxiv.org/abs/2304.14108) for more details.


## xlarge scale models

* `datacomp_xl_s13b_b90k`: A ViT-L/14 trained on DataComp-1B for 12.8B steps and batch size 90k. Achieves 79.2% zero-shot accuracy on ImageNet. Available at https://huggingface.co/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K. 

* `commonpool_xl_clip_s13b_b90k`: A ViT-L/14 trained on CommonPool-XL filtered using CLIP scores, for 12.8B steps and batch size 90k. Achieves 76.4% zero-shot accuracy on ImageNet. Available at https://huggingface.co/laion/CLIP-ViT-L-14-CommonPool.XL.clip-s13B-b90K.

* `commonpool_xl_laion_s13b_b90k`: A ViT-L/14 trained on CommonPool-XL filtered using the LAION-2B filtering scheme, for 12.8B steps and batch size 90k. Achieves 75.5% zero-shot accuracy on ImageNet. Available at https://huggingface.co/laion/CLIP-ViT-L-14-CommonPool.XL.laion-s13B-b90K.

* `commonpool_xl_s13b_b90k`: A ViT-L/14 trained on CommonPool-XL without any filtering, for 12.8B steps and batch size 90k. Achieves 72.3% zero-shot accuracy on ImageNet. Available at https://huggingface.co/laion/CLIP-ViT-L-14-CommonPool.XL-s13B-b90K.


## large scale models

* `datacomp_l_s1b_b8k`: A ViT-B/16 trained on a 140M subset of DataComp-1B, for 1.28B steps and batch size 8k. Achieves 63.1% zero-shot accuracy on ImageNet. Available at https://huggingface.co/laion/CLIP-ViT-B-16-DataComp.L-s1B-b8K.

* `commonpool_l_clip_s1b_b8k`: A ViT-B/16 trained on CommonPool-L filtered using CLIP scores, for 1.28B steps and batch size 8k. Achieves 57.8% zero-shot accuracy on ImageNet. Available at https://huggingface.co/laion/CLIP-ViT-B-16-CommonPool.L.clip-s1B-b8K.

* `commonpool_l_laion_s1b_b8k`: A ViT-B/16 trained on CommonPool-L filtered using the LAION-2B filtering scheme, for 1.28B steps and batch size 8k. Achieves 55.3% zero-shot accuracy on ImageNet. Available at https://huggingface.co/laion/CLIP-ViT-B-16-CommonPool.L.laion-s1B-b8K.

* `commonpool_l_image_s1b_b8k`: A ViT-B/16 trained on CommonPool-L filtered using image-based filtering, for 1.28B steps and batch size 8k. Achieves 57.2% zero-shot accuracy on ImageNet. Available at https://huggingface.co/laion/CLIP-ViT-B-16-CommonPool.L.image-s1B-b8K.

* `commonpool_l_text_s1b_b8k`: A ViT-B/16 trained on CommonPool-L filtered using text-based filtering, for 1.28B steps and batch size 8k. Achieves 56.1% zero-shot accuracy on ImageNet. Available at https://huggingface.co/laion/CLIP-ViT-B-16-CommonPool.L.text-s1B-b8K.

* `commonpool_l_basic_s1b_b8k`: A ViT-B/16 trained on CommonPool-L filtered using basic filtering (English filtering + caption length and image size filtering), for 1.28B steps and batch size 8k. Achieves 51.6% zero-shot accuracy on ImageNet. Available at https://huggingface.co/laion/CLIP-ViT-B-16-CommonPool.L.basic-s1B-b8K.

* `commonpool_l_s1b_b8k`: A ViT-B/16 trained on CommonPool-L without any filtering, for 1.28B steps and batch size 8k. Achieves 45.9% zero-shot accuracy on ImageNet. Available at https://huggingface.co/laion/CLIP-ViT-B-16-CommonPool.L-s1B-b8K.


## medium scale models

* `datacomp_m_s128m_b4k`: A ViT-B/32 trained on a 14M subset of DataComp-1B, for 128M steps and batch size 4k. Achieves 29.7% zero-shot accuracy on ImageNet. Available at https://huggingface.co/laion/CLIP-ViT-B-32-DataComp.M-s128M-b4K.

* `commonpool_m_clip_s128m_b4k`: A ViT-B/32 trained on CommonPool-M filtered using CLIP scores, for 128M steps and batch size 4k. Achieves 27.3% zero-shot accuracy on ImageNet. Available at https://huggingface.co/laion/CLIP-ViT-B-32-CommonPool.M.clip-s128M-b4K.

* `commonpool_m_laion_s128m_b4k`: A ViT-B/32 trained on CommonPool-M filtered using the LAION-2B filtering scheme, for 128M steps and batch size 4k. Achieves 23.0% zero-shot accuracy on ImageNet. Available at https://huggingface.co/laion/CLIP-ViT-B-32-CommonPool.M.laion-s128M-b4K.

* `commonpool_m_image_s128m_b4k`: A ViT-B/32 trained on CommonPool-M filtered using image-based filtering, for 128M steps and batch size 4k. Achieves 26.8% zero-shot accuracy on ImageNet. Available at https://huggingface.co/laion/CLIP-ViT-B-32-CommonPool.M.image-s128M-b4K.

* `commonpool_m_text_s128m_b4k`:  A ViT-B/32 trained on CommonPool-M filtered using text-based filtering, for 128M steps and batch size 4k. Achieves 25.5% zero-shot accuracy on ImageNet. Available at https://huggingface.co/laion/CLIP-ViT-B-32-CommonPool.M.text-s128M-b4K.

* `commonpool_m_basic_s128m_b4k`:  A ViT-B/32 trained on CommonPool-M filtered using basic filtering (English filtering + caption length and image size filtering), for 128M steps and batch size 4k. Achieves 22.6% zero-shot accuracy on ImageNet. Available at https://huggingface.co/laion/CLIP-ViT-B-32-CommonPool.M.basic-s128M-b4K.

* `commonpool_m_s128m_b4k`: A ViT-B/32 trained on CommonPool-M without any filtering, for 128M steps and batch size 4k. Achieves 17.6% zero-shot accuracy on ImageNet. Available at https://huggingface.co/laion/CLIP-ViT-B-32-CommonPool.M-s128M-b4K.


## small scale models

* `datacomp_s_s13m_b4k`: A ViT-B/32 trained on a 1.4M subset of DataComp-1B, for 12.8M steps and batch size 4k. Achieves 3.9% zero-shot accuracy on ImageNet. Available at https://huggingface.co/laion/CLIP-ViT-B-32-DataComp.S-s13M-b4K.

* `commonpool_s_clip_s13m_b4k`: A ViT-B/32 trained on CommonPool-S filtered using CLIP scores, for 12.8M steps and batch size 4k. Achieves 5.1% zero-shot accuracy on ImageNet. Available at https://huggingface.co/laion/CLIP-ViT-B-32-CommonPool.S.clip-s13M-b4K.

* `commonpool_s_laion_s13m_b4k`: A ViT-B/32 trained on CommonPool-S filtered using the LAION-2B filtering scheme scores, for 12.8M steps and batch size 4k. Achieves 3.1% zero-shot accuracy on ImageNet. Available at https://huggingface.co/laion/CLIP-ViT-B-32-CommonPool.S.laion-s13M-b4K.

* `commonpool_s_image_s13m_b4k`: A ViT-B/32 trained on CommonPool-S filtered using image-based filtering, for 12.8M steps and batch size 4k. Achieves 4.3% zero-shot accuracy on ImageNet. Available at https://huggingface.co/laion/CLIP-ViT-B-32-CommonPool.S.image-s13M-b4K.

* `commonpool_s_text_s13m_b4k`: A ViT-B/32 trained on CommonPool-S filtered using text-based filtering, for 12.8M steps and batch size 4k. Achieves 4.6% zero-shot accuracy on ImageNet. Available at https://huggingface.co/laion/CLIP-ViT-B-32-CommonPool.S.text-s13M-b4K.

* `commonpool_s_basic_s13m_b4k`: A ViT-B/32 trained on CommonPool-S filtered using basic filtering (English filtering + caption length and image size filtering), for 12.8M steps and batch size 4k. Achieves 3.0% zero-shot accuracy on ImageNet. Available at https://huggingface.co/laion/CLIP-ViT-B-32-CommonPool.S.basic-s13M-b4K.

* `commonpool_s_s13m_b4k`: A ViT-B/32 trained on CommonPool-S without any filtering, for 12.8M steps and batch size 4k. Achieves 2.5% zero-shot accuracy on ImageNet. Available at https://huggingface.co/laion/CLIP-ViT-B-32-CommonPool.S-s13M-b4K.

