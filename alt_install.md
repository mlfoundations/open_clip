## Installation

1. git clone repo, enter the directory
2. conda env create --file environment.yml
3. conda activate open_clip
4. pip install torch==1.9.0+cu111 torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html
5. pip install git+https://github.com/modestyachts/ImageNetV2_pytorch
6. python setup.py install
7. export PYTHONPATH="$PYTHONPATH:$PWD/src"