import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):  # 倒残差结构
    # expand + depthwise + pointwise

    def __init__(self, in_planes, out_planes, expansion, stride):
        # stride针对的是depthwise的步幅，如果步幅是2则没有shortcut
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes  # 隐藏扩展层：在输入depthwise卷积之前先进行扩展，expansion是扩展率
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1,
                               stride=1, padding=0, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,  # depthwise卷积
                               stride=stride, padding=1, groups=planes, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1,   # pointwise卷积
                               stride=1, padding=0, bias=False)

        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:  # 假设
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))     # 没有采用ReLU非线性激活函数相当于已经采用了线性激活函数
        out = out + self.shortcut(x) if self.stride == 1 else out   # 当深度卷积stride=1的时候采用shortcut
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1, 16, 1, 1),
           (6, 24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6, 32, 3, 2),
           (6, 64, 4, 2),
           (6, 96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)
           ]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1,  # 第一层全卷积输出通道为32
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.layers = self._make_layers(in_planes=32)   # 7层倒残差结构

        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1,
                               padding=0, bias=False)   # 在倒残差之后跟着的点卷积层将通道数扩展到1280层
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:  # 从cfg当中读取layer信息
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                layers.append(
                    Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)  # 完成1280点卷积之后进行平均池化
        out = out.view(out.size(0), -1) # 池化之后将map扁平化
        out = self.linear(out)
        return out


def test():
    net = MobileNetV2()
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())
