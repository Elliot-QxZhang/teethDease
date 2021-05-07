# 中期主要进展

![](asset/flow.png)

**项目进展图：绿色：已完成，红色：产生的问题；黄色：正在实施；蓝色：已确定思路即将实施**

### 1. Web+WeChat+公开数据集制作

### 1.1 Web+weChat小程序开发

我们基于目标检测模型+分类模型+语义分割模型实现了一个端到端的智齿目标检测以及龋块分割标识功能，现已将最高精度模型部署至Web，客户端轻型模型的部署和嵌入式开发仍在进行中，web包括医生端口以及病患端口主要功能为端到端图像检测，辅助功能包括病人病历列表，医生任务列表，医生任务to-do list如图1所示Web展示视频于，微信小程序暂时只设计了demo如图2. 所示

![](asset/web.png)



![](asset/wechat.png)

### 1.2 公开数据集的以及医学数据库制作

目前由于医学道德问题导致的图像收集难度激增，相关公开牙科图像数据集储备几乎为0，为了获取项目训练数据，我们共获得包括温州附属第二医院，蒙恩牙科诊所，温州医学院在内的总计五家医院的助力，共获得20000张牙科全景X光片，部分院方授权书以及全景片样本于 https://github.com/Eric-jinkens/teethDease/tree/main/sample可见，但遗憾的是目前科研界对于医学道德伦理问题的重视，如果需要制作基准公开数据集，我们需要收集患者的知情同意书，目前多家医院正在帮助我们进行收集，授权书模板于 https://github.com/Ericjinkens/teethDease/blob/main/sample/可见

我们为公开数据集制作了医学生学习版本，辅助以一个简单的思维导图用于展示示例，思维导图结构如图3. 所示，但由于医学道德委员会的警告我们不得在未取得患者全部同意书的情况下公开任何图像，所以成型数据集在此处不方便展，后期我们将基于思维导图建立一个关于龋齿疾病的知识图谱，目的在于辅助医学生进行更好的学习示（以下样本全部为已取得同意书的部分图片中处理获得）

![](asset/node.png)

分类数据集中暂时包含2个标签（Caries，Health），每张图像当中包含一个完整的牙齿，原始数据总量为3000，Caries以及Health标签图像各自均有1500。图像格式为TIF，全部为RGB三通道图像，图像大小均调整为224×224，如图4. 所示

![](asset/tooth1.png)

目标检测数据集中包含2个标签(Tooth, Wisdom)，分别代表牙齿以及阻生齿，数据集共包含图片630张，图片格式为TIF，全部为RGB三通道图像，图片大小3048×1372。Tooth标签个数8500，Wisdom标签个数300；原始图像和标注图像如图5. 所示

![](asset/tooth2.png)

根据龋病进展累积牙釉质、牙本质、牙髓分为主要三类以及继发龋，具体表现如图6. 所示：

![](asset/tooth3.png)

### 2. 目标检测模型进展

目标检测方面采用Faster R-CNN模型，利用不同的backbone进行试验，并且尝试在backbone上加入ECA-net注意力模块。目标检测的Model Zoo以及部分performance如下：

![](asset/model_zoo.png)

其中模型Precision的取值均为至少连续10次迭代的Train loss波动幅度小于0.01时停止迭代后最后一个精度值

模型训练方法参数以及训练设备如下：

![](asset/train_method.png)

训练过程中各模型mean loss如下图所示：

![](asset/mean_losses.jpg)

各个模型在验证集上的召回率与AP：

![](asset/recall_ap.png)



为了更近一步地测试模型在召回，准确率以及平均精度上的性能，我们分别绘制了Recall，Precision，AP vs.IoU曲线（右下角为加入ECAnet通道注意力机制）：

![](E:\repo\teethDease\asset\iou.png)

### 3. 分类模型进展

**主要目标：训练出服务器端模型（提升精度不计计算量）和客户端模型（牺牲部分精度减少计算量）**

PS.目前包括自行构建CvT-13，CvT-21的代码以及模型参数pth文件已全部开源

实验思路：

1. 由于对模型计算量的需求不同，我们将模型进行分类：Light，Middle，Heavy，如Table 1.所示

2. 我们搭建了当前轻型CNN的SOTA：GhostNet，并将论文中的GhostModule插入到了VGG系列和ResNet系列中并进行了对比试验，简要结果如Table 1.和图2-1所示，可见对于我们的数据集GhostModule适应的非常好，当前对于我们的数据集轻型模型SOTA为Ghost-Resnet-20

   ​																		            **Table 1. Experiment Model Zoo**

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

