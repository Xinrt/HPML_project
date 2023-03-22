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



## TVM & AutoML

1. TVM是一个可用于高性能机器学习的编译器和运行时库，可以将深度学习模型编译为本地机器代码，从而提高执行效率。在高性能机器学习层面，可以对TVM进行以下方面的研究：

   1. 自动化编译优化：TVM可以进行编译优化，以提高深度学习模型的执行效率。在高性能机器学习中，可以通过研究自动化编译优化算法和策略，进一步提高TVM的性能和效率。

   1. 跨平台优化：TVM支持多种硬件平台，可以在不同的硬件平台上进行高效的部署和执行。在高性能机器学习中，可以通过研究跨平台优化算法和策略，进一步提高TVM的跨平台性能和效率。

​	[不太了解自动化编译优化算法和策略(◞‸◟ ) ]

2. AutoML是一种自动化机器学习的技术，能够自动选择和调整机器学习算法和超参数等，以提高模型性能和准确率。在高性能机器学习层面，可以对AutoML进行以下方面的研究：
   1. 高效搜索算法：AutoML通常需要在大量算法和超参数组合中进行搜索，这个过程可以非常耗时。在高性能机器学习中，可以通过研究高效的搜索算法和策略，提高搜索效率，减少搜索时间。
   2. 自动化优化算法：在高性能机器学习中，可以通过研究自动化优化算法，使得AutoML算法可以快速适应新的硬件和软件环境，提高自动化优化的效率和精度。[不太懂这个是什么意思(◞‸◟ ) ]
   3. 高性能并行计算：在AutoML过程中需要进行大量的训练和评估操作，这个过程可以通过高性能并行计算来加速。在高性能机器学习中，可以研究高性能并行计算算法和技术，以提高AutoML的效率和速度。
3. AutoML+TVM



