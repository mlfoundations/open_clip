#!/bin/bash

args=''
for i in "$@"; do 
  i="${i//\\/\\\\}"
  args="$args \"${i//\"/\\\"}\""
done
echo $args
ls
if [ "$args" == "" ]; then args="/bin/bash"; fi

if [[ "$(hostname -s)" =~ ^g[r,v,a] ]]; then nv="--nv"; fi

singularity \
  exec $nv \
  --overlay /scratch/bf996/singularity_containers/openclip_env_cuda.ext3:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
  /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
  /bin/bash -c "
 source /ext3/env.sh
 $args 
"