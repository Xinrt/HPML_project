
# HPML_project

## Stitch ResNet-18 to ResNet-34

### Recur SN-Net stitch on ResNet

#### Training

1. First, install the dependencies

```
conda create -p /scratch/$USER snnet_env python=3.9
conda activate /scratch/$USER/snnet_env
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.--extra-index-url https://download.pytorch.org/whl/cu113 
pip install fvcore
pip install timm==0.6.12
```

2. Download the imagenet, put it in a directory you desire

3. Then, run the following command, notice that currently only 1 GPU could be used

```
./distributed_train.sh 1 \
[path/to/imagenet] \
-b 128 \
--stitch_config configs/resnet18_resnet50.json \
--sched cosine \
--epochs 30 \
--lr 0.05 \
--amp --remode pixel \
--reprob 0.6 \
--aa rand-m9-mstd0.5-inc1 \
--resplit --split-bn -j 10 --dist-bn reduce
```

## reference

### **Stitchable Neural Networks**

<https://arxiv.org/abs/2302.06586>

<https://github.com/ziplab/SN-Net>

利用现有的model family直接做少量epoch finetune就可以得到大量插值般存在的子网络，运行时任意切换网络结构满足不同resource constraint -> 用少量计算资源满足目标场景

[把SN-Net extend到其他任务上：such as natural language processing, dense prediction and transfer learning]

更多拓展空间：

1. 当前的训练策略比较简单，每次iteration sample出来一个stitch，但是当stitches特别多的时候，可能导致某些stitch训练的不够充分，除非增加训练时间。所以训练策略上可以继续改进。
2. anchor的performance会比之前下降一些，虽然不大。直觉上，在joint training过程中，anchor为了保证众多stitches的性能在自身weights上做了一些trade-off。目前补充材料里发现finetune更多epoch可以把这部分损失补回来。
3. 不用nearest stitching可以明显扩大space，但此时大部分网络不在pareto frontier上，未来可以结合训练策略进行改进，或者在其他地方发现advantage。



### ResNets

**ResNet-18**: https://huggingface.co/microsoft/resnet-18

**ResNet-34**: https://huggingface.co/microsoft/resnet-34





## optimize algorithm in cuda

<https://github.com/BBuf/how-to-optim-algorithm-in-cuda>



## use PyTorch 2.0

https://mp.weixin.qq.com/s/BFnapl7TpPkpa7Dm9sC8BQ



## overleaf

**Proposal**:

<https://www.overleaf.com/5277458364tsvsnmmpftms>
