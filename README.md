# HPML_project

## description of the project











## description of the repository and code structure





## Environment setup on NYU HPC 

### Training using Singularity - on burst compute

#### Prepare environment

Let's start

```shell
ssh <NetID>@greene.hpc.nyu.edu
```

ssh to the class on GCP (burst login node) - anyone can login but you can only submit jobs if you have approval

```shell
ssh burst
```

Start an interactive job

```shell
srun --cpus-per-task=8 --time=4:00:00 --mem=20GB --account=ece_gy_9143-2023sp --partition=n1s16-v100-2 --gres=gpu:2 --pty /bin/bash
```



Create a directory for the environment

```shell
mkdir /scratch/<NetID>/snnet_env_burst
cd /scratch/<NetID>/snnet_env_burst
```

Copy an appropriate gzipped overlay images from the overlay directory. 

```shell
ls /share/apps/overlay-fs-ext3
```

In this example we use overlay-15GB-500K.ext3.gz as it has enough available storage for most conda environments. It has 15GB free space inside and is able to hold 500K files

```Shell
cp -rp /share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz .
gunzip overlay-15GB-500K.ext3.gz
```

Launch the appropriate Singularity container in read/write mode (with the :rw flag)

```shell
singularity exec --overlay overlay-15GB-500K.ext3:rw /share/apps/images/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash
```

The above starts a bash shell inside the referenced Singularity Container overlayed with the 15GB 500K you set up earlier. This creates the functional illusion of having a writable filesystem inside the typically read-only Singularity container.

Now, inside the container, download and install miniconda to /ext3/miniconda3

```Shell
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /ext3/miniconda3
# rm Miniconda3-latest-Linux-x86_64.sh # if you don't need this file any longer
```

Next, create a wrapper script /ext3/env.sh

The wrapper script will activate your conda environment, to which you will be installing your packages and dependencies. The script should contain the following:

```Shell
#!/bin/bash

source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH
export PYTHONPATH=/ext3/miniconda3/bin:$PATH
```

Activate conda environment with the following:

```Shell
source /ext3/env.sh
```

update and install packages

```shell
conda update -n base conda -y
conda clean --all --yes
conda install pip -y
conda install ipykernel -y # Note: ipykernel is required to run as a kernel in the Open OnDemand Jupyter Notebooks
```

To confirm that your environment is appropriately referencing your Miniconda installation, try out the following:

```Shell
which conda
# output: /ext3/miniconda3/bin/conda

which python
# output: /ext3/miniconda3/bin/python

python --version
# output: Python 3.8.5

which pip
# output: /ext3/miniconda3/bin/pip

```

install the dependencies

```shell
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 
pip install fvcore
pip install timm==0.6.12
```

```Shell
pip list
# check: torch, fvcore, timm
```

```Shell
exit
# exit Singularity
```



#### Training 

##### Using command line

1. First, launch Singularity with mouting scratch path

```Shell
singularity shell -B /scratch/<Net-id> --nv --overlay /scratch/<Net-id>/snnet_env_burst/overlay-15GB-500K.ext3:ro /share/apps/images/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif

source /ext3/env.sh

which python
#check output: /ext3/miniconda3/bin/python
```

2. Download the imagenet, put it in a directory you desire

3. Then, run the following command, notice that currently 2 GPU could be used

   **remenber to specify the output path, and creat the corresponding directory**

```Shell
cd /scratch/<Net-id>/SN-Net/stitching_resnet_swin

./distributed_train.sh 2 \
[path/to/imagenet] \
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
```



##### Using sbatch

put the `compute_run.sbatch` at the same location of `train.py`, in burst login mode

the slurm output will display in burst compute mode at the same location of `train.py`





GPUs avaliable on HPC:
https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/best-practices#h.ud1g8fsehnmk



## Example commands to execute the code 

```Shell
cd /scratch/<Net-id>/
ssh burst
srun --cpus-per-task=8 --time=4:00:00 --mem=20GB --account=ece_gy_9143-2023sp --partition=n1c24m128-v100-4 --gres=gpu:4 --pty /bin/bash

cd /scratch/<Net-id>/snnet_env_burst
singularity shell -B /scratch/<Net-id> --nv --overlay /scratch/<Net-id>/snnet_env_burst/overlay-15GB-500K.ext3:ro /share/apps/images/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif

source /ext3/env.sh

cd /scratch/<Net-id>
git clone https://github.com/Xinrt/HPML_project.git 
cd /HPML_project/project

./distributed_train.sh 4 \
[path/to/imagenet] \
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
```





## Results (including charts/tables) and observations 







## reference

paper link: <https://arxiv.org/abs/2302.06586>

github link: <https://github.com/ziplab/SN-Net>

