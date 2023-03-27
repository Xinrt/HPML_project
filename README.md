

# HPML_project

## reference
1. ATCON: Attention Consistency for Vision Models
     link: https://github.com/alimirzazadeh/semisupervisedattention
       注意力机制是指在神经网络中，模型可以选择性地将其关注点放在输入数据的某些部分，以此来提高模型的准确性和性能。

2. Batch Norm层
     link: https://github.com/mr-eggplant/SAR
       batch 无关的 Norm 层（Group 和 Layer Norm）一定程度上规避了 Batch Norm 局限性，更适合在动态开放场景中执行 TTA，其稳定性也更高

3. Dense QR
     使用QR分解权重矩阵的主要目的是减少神经网络中的参数数量和计算复杂度，从而提高神经网络的训练效率和泛化能力

4. BiFormer: Vision Transformer with Bi-Level Routing Attention

     https://github.com/rayleizhu/BiFormer

  原始的 Transformer 架构设计中，这种结构虽然在一定程度上带来了性能上的提升，但却会引起两个老生常态的问题：

  1. **内存占用大**
  2. **计算代价高**

  因此，有许多研究也在致力于做一些这方面的优化工作，包括但不仅限于将注意力操作限制在：

  - `inside local windows`, e.g., `Swin transformer` and `Crossformer`;
  - `axial stripes`, e.g., `Cswin transformer`;
  - `dilated windows`, e.g., `Maxvit` and `Crossformer`;

  作者认为以上这些方法大都是通过将 手工制作 和 与内容无关 的稀疏性引入到注意力机制来试图缓解这个问题。因此，本文通过双层路由(`bi-level routing`)提出了一种新颖的**动态稀疏注意力**(`dynamic sparse attention `)，以实现更灵活的**计算分配**和**内容感知**，使其具备动态的查询感知稀疏性，

  -> 基于该基础模块，本文构建了一个名为`BiFormer`的新型通用视觉网络架构。由于 BiFormer 以查询自适应的方式关注一小部分相关标记，而不会分散其他不相关标记的注意力，因此它具有良好的性能和高计算效率。

  -> 通过在图像分类、目标检测和语义分割等多项计算机视觉任务的实证结果充分验证了所提方法的有效性

  ->本文方法貌似对小目标检测效果比较好。这可能是因为`BRA`模块是基于稀疏采样而不是下采样，一来可以保留细粒度的细节信息，二来同样可以达到节省计算量的目的。

  [感觉像是升级版的transformer]

  Transformer 是一种用于处理序列数据的模型，使用了自注意力机制（Self-Attention Mechanism）来建模序列之间的依赖关系，主要应用于自然语言处理领域，如机器翻译、文本分类、语言建模等任务

  [我觉得这篇文章提出的BiFormer模型就是让transformer能应用在计算机视觉领域]



​		这篇文章已经对比了BiFormer和其他计算机视觉算法在image classification (Sec. 4.1), object detection and instance segmentation (Sec. 4.2), and semantic segmentation (Sec. 4.3)的准确度



​		**我们可不可以比较速度，内存啥的？**

5. 基于CLIP的微调新范式（cross-modal adaptation)

   https://github.com/linzhiqiu/cross_modal_adaptation

   https://arxiv.org/abs/2301.06267

   https://linzhiqiu.github.io/papers/cross_modal/

   仅用线性分类器即可超越CoOp，Tip-Adapter等多种算法在小样本图像识别训练集上的性能

​		[量化研究跨模态微调（cross-modal adaptation）能不能取代单模态微调，成为未来预训练模型的性能衡量基准？]



6. Stitchable Neural Networks

   https://arxiv.org/abs/2302.06586

   https://github.com/ziplab/SN-Net

   利用现有的model family直接做少量epoch finetune就可以得到大量插值般存在的子网络，运行时任意切换网络结构满足不同resource constraint -> 用少量计算资源满足目标场景

   [把SN-Net extend到其他任务上：such as natural language processing, dense prediction and transfer learning]

   更多拓展空间：

   1. 当前的训练策略比较简单，每次iteration sample出来一个stitch，但是当stitches特别多的时候，可能导致某些stitch训练的不够充分，除非增加训练时间。所以训练策略上可以继续改进。
   2. anchor的performance会比之前下降一些，虽然不大。直觉上，在joint training过程中，anchor为了保证众多stitches的性能在自身weights上做了一些trade-off。目前补充材料里发现finetune更多epoch可以把这部分损失补回来。
   3. 不用nearest stitching可以明显扩大space，但此时大部分网络不在pareto frontier上，未来可以结合训练策略进行改进，或者在其他地方发现advantage。





## 方向

1. **Comprehensive Review of Deep Learning-Based 3D Point Cloud Completion Processing and Analysis**

   https://arxiv.org/abs/2203.03311

   **可做：**Part: IX. FUTURE DIRECTION AND OPEN QUESTION

2. **Enable Deep Learning on Mobile Devices: Methods, Systems, and Applications**

   https://dl.acm.org/doi/pdf/10.1145/3486618

   **可做：**future direction部分

3. **Reinforcement Learning with Neural Radiance Fields**

   https://arxiv.org/abs/2206.01634

   **可做：**investigate NeRF supervision in an online setup



## optimize algorithm in cuda
https://github.com/BBuf/how-to-optim-algorithm-in-cuda



## overleaf

https://www.overleaf.com/5277458364tsvsnmmpftms
