import torch
from torch import nn


class Net(nn.Module):
    # input_channel, output_channel, stride
    cfg = [
        (32, 64, 1),
        (64, 128, 2),
        (128, 128, 1),
        (128, 256, 2),
        (256, 256, 1),
        (256, 512, 2),
        (512, 512, 1),
        (512, 512, 1),
        (512, 512, 1),
        (512, 512, 1),
        (512, 512, 1),
        (512, 1024, 2),
        (1024, 1024, 1)
    ]

    def __init__(self):
        super(Net, self).__init__()

        def conv_bn(input_channel, output_channel, stride):  # full_conv层，MobileNet除了第一层以外都是深度可分离卷积
            return nn.Sequential(
                nn.Conv2d(input_channel, output_channel, 3, stride, 1, bias=False),
                nn.BatchNorm2d(output_channel),
                nn.ReLU(inplace=True)
            )

        def conv_dw(input_channel, output_channel, stride):  # 深度可分离卷积
            return nn.Sequential(
                # depth wise卷积：group的目的指的是将输入通道进行分组：默认分为一组（分组指的是一组之内会进行相加），分为input_channel组就是不进行相加）
                # group如果为2，kernel在通道维上还是和输入通道数相同，但是会分为两组进行相加，kernel在第四维度上就是输出通道/2
                # MobileNet
                nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=3,
                          stride=stride, padding=1, groups=input_channel, bias=False),

                nn.BatchNorm2d(input_channel),
                nn.ReLU(inplace=True),

                nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False),
                nn.BatchNorm2d(output_channel),
                nn.ReLU(inplace=True),  # 在V2当中此处的激活函数变更为线性激活函数
            )

        def _make_layers(self: nn.Module):
            layers = [conv_bn(3, 32, 2)]
            for input_channel, output_channel, stride in self.cfg:  # 从cfg当中读取layer信息
                layers.append(
                    conv_dw(input_channel, output_channel, stride))
            layers.append(nn.AvgPool2d(7))
            return nn.Sequential(*layers)

        self.model = _make_layers(self)
        self.fc = nn.Linear(1024, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.model(x)
        # 这一步相当于只FlattenLayer层,将批量大小保留，然后转为二维向量
        x = x.view(x.shape[0], -1)
        # softmax的计算会被归入损失函数的交叉损失熵当中
        x = self.fc(x)
        x = self.fc2(x)
        return x


test = torch.rand(1, 3, 224, 224).to("cuda")
net = Net().to("cuda")
print(net(test))
