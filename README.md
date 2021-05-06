# 中期主要进展
**主要目标：训练出服务器端模型（提升精度不计计算量）和客户端模型（牺牲部分精度减少计算量）**

PS.目前包括自行构建CvT-13，CvT-21的代码以及模型参数pth文件已全部开源

实验思路：

1. 由于对模型计算量的需求不同，我们将模型进行分类：Light，Middle，Heavy，如Table 1.所示

2. 我们搭建了当前轻型CNN的SOTA：GhostNet，并将论文中的GhostModule插入到了VGG系列和ResNet系列中并进行了对比试验，简要结果如Table 1.和图2-1所示，可见对于我们的数据集GhostModule适应的非常好，当前对于我们的数据集轻型模型SOTA为Ghost-Resnet-20

   ​																		**Table 1. Experiment Model Zoo**

   | Model              | Acc.(%) | Category     | Weights(M) |
   | ------------------ | ------- | ------------ | ---------- |
   | GhostNet 0.5×      | 82.1    | Light CNN    | 1.3        |
   | GhostNet 1.0×      | 83.5    | Light CNN    | 3.9        |
   | GhostNet 1.5×      | 86.5    | Light CNN    | 7.8        |
   | MobileNetV3  Small | 81.5    | Light CNN    | 1.5        |
   | MobileNetV3  Large | 82.3    | Light CNN    | 4.2        |
   | GoogLeNet          | 89.3    | Light CNN    | 6.6        |
   | ResNet-20          | 89.2    | Middle CNN   | 11.2       |
   | ResNet-32          | 89.1    | Middle CNN   | 21.3       |
   | ResNet-44          | 89.5    | Middle CNN   | 23.5       |
   | ResNet-56          | 89.4    | Middle CNN   | 27.8       |
   | VGG-11             | 90.2    | Heavy  CNN   | 132.9      |
   | VGG-13             | 89.7    | Heavy  CNN   | 133.0      |
   | VGG-16             | 90.1    | Heavy  CNN   | 138.4      |
   | VGG-19             | 89.4    | Heavy  CNN   | 143.7      |
   | Ghost-VGG-11       | 90.7    | Light CNN    | 4.7        |
   | Ghost-VGG-13       | 90.3    | Light CNN    | 4.8        |
   | Ghost-VGG-16       | 90.2    | Light CNN    | 15.0       |
   | Ghost-VGG-19       | 90.3    | Middle CNN   | 10.2       |
   | Ghost-ResNet-20    | 90.7    | EX Light CNN | 0.14       |
   | Ghost-ResNet-32    | 89.1    | EX Light CNN | 0.24       |
   | Ghost-ResNet-44    | 90.1    | EX Light CNN | 0.34       |
   | Ghost-ResNet-56    | 90.2    | EX Light CNN | 0.45       |
   | ViT                | 88.6    | Transformer  | 53.5       |
   | TNT                | 88.3    | Transformer  | 20.4       |
   | CvT-13             | 91.1    | Transformer  | 19.98      |
   | CvT-21             | 90.2    | Transformer  | 31.54      |

3. 针对服务器端模型，除了当前主流的CNN模型之外，考虑到Transformer在全局建模能力上的优秀，我们借鉴了”**CvT:    Introducing Convolutions to Vision Transformers”**，当前其官方开源代码仍未发布，我们利用pytorch自行进行复现，代码已上传GitHub，我们将在论文作者的官方代码开源后重新进行实验，精度如Table 1.所示(蓝色为client model SOTA，红色为server model SOTA），与CNN SOTA对比试验如图2-2所示

   ![](asset/accvsepoch.bmp)

   对比试验采用K=10的K折交叉折叠验证，优化算法采用Adam，初始学习率设置为0.0001，并采用余弦退火法来保证loss的平滑下降，对比试验采用华为云p100GPU，最大迭代次数设置为500，如Table 2. 所示

   ​                                             **Table 2. Localization network configuration of classification model** 

   | Training Option           | Value             |
   | ------------------------- | ----------------- |
   | Optimization Method       | Adam              |
   | Mini-Batch Size           | 128               |
   | K-Folder Cross Validation | K=10              |
   | Initial Learning Rate     | 1e-3              |
   | Learning Rate Schedule    | CosineAnnealingLR |
   | ETA-min                   | 0.00001           |
   | Max Epochs                | 500               |
   | Training Environment      | GPU 1×p100        |
   | Framework                 | PyTorch           |

4. 由于医学疾病检测任务的特殊性，单一使用Accuracy并不能完整的评判模型的优劣，我们对部分的模型的在验证集上的Recall（查全率），precision（查准率），PR曲线进行了分析对比，分别如Table 3. 图3.所示

   ​															**图3. Precision v.s. Recall of some classification models**

   ![](asset/PR.bmp)

   根据来自温州附属第二医院口腔科主任陈丽芬女士的建议，医生们更加倾向于模型在判断时更加的激进即更倾向于将图片判断为正样例，我们首先在验证集上画出了混淆矩阵，如图4. 所示

   ![](asset/confusion_matrix.bmp)

   由混淆矩阵可见仍旧存在部分将龋齿误判为健康齿的可能性，随机我们调节了交叉损失熵函数中的class_weight,**将caries类别的weight调节为1.6之后**重新进行训练，所有模型的FN降低到了3%以下

5. **由于目标检测模型的performance上达到了非常高的水平，意味着我们不需要标注全景片就可以获得非常大量的无标签牙齿图像，我们引入了Mean Teacher半监督模型并借鉴了Semi-supervised Medical Image Classification with Relation-driven Self-ensembling Model”引入了sample relation consistency(SRC)进一步提取无标签样本的语义信息，我们基于论文自己实现了带有SRC的MT半监督模型，并将论文中作为backbone的densenet121更换为表现更加优秀的CvT-13，具体结构如图5.所示，成就了我们数据集上的state of art，最终精度达到了94.7%**

   ![](asset/ss-lrc-CvT.png)

