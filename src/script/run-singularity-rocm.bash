#!/bin/bash

args=''
for i in "$@"; do 
  i="${i//\\/\\\\}"
  args="$args \"${i//\"/\\\"}\""
done
echo $args
ls
if [ "$args" == "" ]; then args="/bin/bash"; fi

singularity \
  exec --rocm \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
  /scratch/work/public/singularity/hudson/images/rocm4.5.2-ubuntu20.04.3.sif \
  /bin/bash -c "
  source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
  conda activate ./rocm_penv;
  pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/rocm4.5.2;
  pip install -r requirements-training.txt;
 $args 
"