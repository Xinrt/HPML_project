#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=20GB
#SBATCH --account=ece_gy_9143-2023sp
#SBATCH --partition=n1s16-v100-2
#SBATCH --gres=gpu:2  ## To request specific GPU (v100 or rtx8000)

cd /scratch/xt2191/snnet_env_burst

singularity shell -B /scratch/xt2191 --nv --overlay /scratch/xt2191/snnet_env_burst/overlay-15GB-500K.ext3:ro /share/apps/images/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif -c "source /ext3/env.sh; cd /scratch/xt2191/SN-Net/stitching_resnet_swin;
 ./distributed_train.sh 2 \
/scratch/xt2191/tiny-imagenet-200 \
-b 128 \
--stitch_config configs/resnet18_resnet50.json \
--output './output/train' \
--sched cosine \
--epochs 30 \
--lr 0.05 \
--amp --remode pixel \
--reprob 0.6 \
--aa rand-m9-mstd0.5-inc1 \
--resplit --split-bn -j 10 --dist-bn reduce
" 
