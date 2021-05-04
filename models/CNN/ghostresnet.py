import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def _weights_init(m):
    classname = m.__class__.__name__

    # print(classname)

    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


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

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):

        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))

            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ghost_res_block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):

        super(ghost_res_block, self).__init__()

        self.conv1 = GhostModule(in_planes, planes, kernel_size=3, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = GhostModule(planes, planes, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))

            elif option == 'B':
                self.shortcut = nn.Sequential(
                    GhostModule(in_planes, self.expansion * planes, kernel_size=1, stride=stride),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = GhostModule(3, 16, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(with_ghost=True):
    if with_ghost:
        return ResNet(ghost_res_block, [3, 3, 3])
    else:
        return ResNet(BasicBlock, [3, 3, 3])


def ghostresnet32(with_ghost=True):
    if with_ghost:
        return ResNet(ghost_res_block, [5, 5, 5])
    else:
        return ResNet(BasicBlock, [5, 5, 5])


def ghostresnet44(with_ghost=True):
    if with_ghost:
        return ResNet(ghost_res_block, [7, 7, 7])
    else:
        return ResNet(BasicBlock, [7, 7, 7])


def ghostresnet56(with_ghost=True):
    if with_ghost:
        return ResNet(ghost_res_block, [9, 9, 9])
    else:
        return ResNet(BasicBlock, [9, 9, 9])


def ghostresnet110(with_ghost=True):
    if with_ghost:
        return ResNet(ghost_res_block, [18, 18, 18])
    else:
        return ResNet(BasicBlock, [200, 200, 200])


def ghostresnet1202(with_ghost=True):
    if with_ghost:
        return ResNet(ghost_res_block, [200, 200, 200])
    else:
        return ResNet(BasicBlock, [200, 200, 200])
