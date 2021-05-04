import math
import torch
import torch.nn as nn
import torch.nn.functional as F

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _make_divisible(v, divisor, min_value=None):  # 保证通道数能够被4整除，设置通道数只需要设置扩展率
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):  # 与MobileNet相同的激活函数,但是并不适用于GhostBottleNeck，而是仅仅适用于SE结构
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class SqueezeExcite(nn.Module):  # SE结构
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn

        # 根据se_ratio计算出reduce层的通道数
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)  # Squeeze层，将通道数先降低到reduce
        self.act1 = act_layer(inplace=True)  # 进行excite操作，利用ReLU进行筛选性激活
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)  # 重新将通道数进行扩展

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)  # x*h_sigmoid相当于h_swish激活函数
        return x


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True, fmap_order=None):
        super(GhostModule, self).__init__()
        self.fmap_order = fmap_order
        self.oup = oup
        init_channels = int(math.ceil(oup / ratio))
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        if isinstance(self.fmap_order, list):
            out_sort = out.clone()
            for i, order in enumerate(self.fmap_order):
                out_sort[:, order, :, :] = out[:, i, :, :]  # eg. fmap_order=[3, 0, 1, 2],  0->3, 1->0, 2->1, 3->2
            out = out_sort
        return out[:, :self.oup, :, :]


class VGG(nn.Module):
    def __init__(self, vgg_name, with_ghost=True):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], with_ghost=with_ghost)
        self.classifier = nn.Linear(25088, 2)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, with_ghost=True):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [SqueezeExcite(in_channels, se_ratio=0.25)]
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if with_ghost:
                    module = GhostModule(inp=in_channels, oup=x, kernel_size=3)
                else:
                    module = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
                layers += [
                    module,
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def create_vgg(model_name):
    return VGG(model_name)
