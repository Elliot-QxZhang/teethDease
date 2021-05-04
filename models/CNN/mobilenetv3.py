import torch
import torch.nn as nn
import math

from thop import profile

__all__ = ['mobilenetv3_large', 'mobilenetv3_small']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):  # h_swish激活函数的部分函数计算
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):  # 新的激活函数h_swish
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):  # SE结构：部分的Bneck存在SE激活层
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(_make_divisible(channel // reduction, 8), channel),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):  # 基本的3*3全卷积结构
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):  # 基本的点卷积结构
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual(nn.Module):  # 新的倒残差结构
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):  # hidden_dim表示经过扩展层之后的通道数目
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]  # 倒残差结构存在stride=1以及stride=2

        self.identity = stride == 1 and inp == oup  # identy为真当且仅当为stride=1的倒残差结构

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw：depth_wise卷积，在这个if当中将不会进行expansion,输入通道等于隐藏扩展后通道
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                          (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),  # 利用h_swish代替ReLU作为新的激活函数

                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),  # use_se标记了这一层是否使用SE结构，nn.Identity()表示直接输出输入

                # pw-linear，重新收缩通道数目
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw：expansion层，将通道数目进行扩展
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),

                # dw：depth_wise卷积
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                          (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, cfgs, mode, num_classes=2, width_mult=1.):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        assert mode in ['large', 'small']  # MobileNet分为两个模式

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]  # 设置第一层全卷积层，与V2不同的是产生的输出通道不是32而是16

        # building inverted residual blocks
        block = InvertedResidual
        for kernel_size, t, c, use_se, use_hs, stride in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)  # 设置expansion层的扩展通道数，t为扩展因子

            # 此处插入了所有的Bneck
            layers.append(block(input_channel, exp_size, output_channel, kernel_size, stride, use_se, use_hs))

            input_channel = output_channel
        self.features = nn.Sequential(*layers)

        # building last several layers
        self.conv = conv_1x1_bn(input_channel, exp_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        output_channel = {'large': 1280, 'small': 1024}  # 根据mode设置最后1*1卷积层的输入通道数目
        output_channel = _make_divisible(output_channel[mode] * width_mult, 8) \
            if width_mult > 1.0 else output_channel[mode]

        self.classifier = nn.Sequential(  # 最后的全连接层，池化层在1*1卷积层之前就已经完成
            nn.Linear(exp_size, output_channel),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv3_large(**kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # kernel_size, t(扩展因子), c（表示输出通道/8）, SE, HS, s
        [3, 1, 16, 0, 0, 1],
        [3, 4, 24, 0, 0, 2],
        [3, 3, 24, 0, 0, 1],
        [5, 3, 40, 1, 0, 2],
        [5, 3, 40, 1, 0, 1],
        [5, 3, 40, 1, 0, 1],
        [3, 6, 80, 0, 1, 2],
        [3, 2.5, 80, 0, 1, 1],
        [3, 2.3, 80, 0, 1, 1],
        [3, 2.3, 80, 0, 1, 1],
        [3, 6, 112, 1, 1, 1],
        [3, 6, 112, 1, 1, 1],
        [5, 6, 160, 1, 1, 2],
        [5, 6, 160, 1, 1, 1],
        [5, 6, 160, 1, 1, 1]
    ]
    return MobileNetV3(cfgs, mode='large', **kwargs)


def mobilenetv3_small(**kwargs):
    """
    Constructs a MobileNetV3-Small model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3, 1, 16, 1, 0, 2],
        [3, 4.5, 24, 0, 0, 2],
        [3, 3.67, 24, 0, 0, 1],
        [5, 4, 40, 1, 1, 2],
        [5, 6, 40, 1, 1, 1],
        [5, 6, 40, 1, 1, 1],
        [5, 3, 48, 1, 1, 1],
        [5, 3, 48, 1, 1, 1],
        [5, 6, 96, 1, 1, 2],
        [5, 6, 96, 1, 1, 1],
        [5, 6, 96, 1, 1, 1],
    ]

    return MobileNetV3(cfgs, mode='small', **kwargs)


model_s = mobilenetv3_small()
model_l = mobilenetv3_large()
input = torch.randn(1, 3, 224, 224)
flops, params = profile(model_s, inputs=(input, ))
flops_l, params_l = profile(model_l, inputs=(input, ))
print('flops:'+str(flops))
print('params'+str(params))
print('flops:'+str(flops_l))
print('params'+str(params_l))