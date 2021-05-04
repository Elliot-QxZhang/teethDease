import os
import sys
import shutil
import argparse
import logging
import time
import random
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid


# 使用benchmark以启动CUDNN_FIND自动寻找最快的操作，当计算图不会改变的时候（每次输入形状相同，模型不改变）的情况下可以提高性能，反之则降低性能
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(1337)
np.random.seed(1337)
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


# 获取
def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 1.0 * sigmoid_rampup(epoch, 30)


# ema指的是老师模型根据学生模型进行调整参数
def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


# dataset-defination
from torch.utils.data import Dataset
from PIL import Image
import itertools
from torch.utils.data.sampler import Sampler

N_CLASSES_TOOTH = 2
CLASS_NAMES_TOOTH = ['Caries', 'Health']


class Tooth_Dataset(Dataset):
    def __init__(self, transform=None):
        super(Tooth_Dataset, self).__init__()
        imgs = []
        labels = []
        labeled = []
        for dirname, _, filenames in os.walk('../input/tooth-project-2/dataset_origin/dataset_origin/caries'):
            for filename in filenames:
                imgs.append(str(os.path.join(dirname, filename)))
                labeled.append(str(os.path.join(dirname, filename)))
                labels.append([0, 1, 0])
        for dirname, _, filenames in os.walk('../input/tooth-project-2/dataset_origin/dataset_origin/health'):
            for filename in filenames:
                imgs.append(str(os.path.join(dirname, filename)))
                labeled.append(str(os.path.join(dirname, filename)))
                labels.append([1, 0, 0])
        # 无标签数据的引入
        for dirname, _, filenames in os.walk('../input/unlabeled/unlabeled'):
            for filename in filenames:
                imgs.append(str(os.path.join(dirname, filename)))
                labels.append([0, 0, 0])
        self.images = imgs
        self.labels = labels
        self.labeled = labeled
        self.transform = transform
        print('labeled images:{}'.format(len(self.labeled)))
        print('Total # images:{}, labels:{}'.format(len(self.images), len(self.labels)))

    def __getitem__(self, index):
        items = self.images[index]
        image = Image.open(items).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        # image中包含了有标签数据和无标签数据
        return items, index, image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.images)


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    # primary_indices为已标注数据的索引的list，secondary_indices为未标注数据的索引的list
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        # primary_iter：已标注数据的索引序列打乱顺序，每次迭代只会遍历一次已标注数据
        primary_iter = iterate_once(self.primary_indices)
        # secondary_iter：未标注数据的索引序列，每次迭代可以遍历很多次
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            # grouper指的是在两个序列当中进行采样
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


def iterate_once(iterable):
    return np.random.permutation(iterable)  # 产生一个随机序列


def iterate_eternally(indices):  # 将多个迭代器进行高效连接，传入的是未标注数据索引的list
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


# end of dataset defination

# defination of model(densenet)
import torch
import torch.nn as nn
import torch.nn.functional as F
# from networks import densenet
import re
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import os


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.drop_layer = nn.Dropout(p=drop_rate)

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        # if self.drop_rate > 0:
        #     print (self.drop_rate)
        #     new_features = self.drop_layer(new_features)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        print(out.size())
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        print(out.size())
        out = self.classifier(out)
        return out


def densenet121(pretrained=False, **kwargs):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), **kwargs)
    return model


class DenseNet121(nn.Module):
    def __init__(self, out_size, mode, drop_rate=0.0):
        super(DenseNet121, self).__init__()
        assert mode in ('U-Ones', 'U-Zeros', 'U-MultiClass')
        self.densenet121 = densenet121(pretrained=False, drop_rate=drop_rate)
        num_ftrs = self.densenet121.classifier.in_features
        # 加上一个全连接层
        if mode in ('U-Ones', 'U-Zeros'):
            self.densenet121.classifier = nn.Sequential(
                nn.Linear(num_ftrs, out_size),
                # nn.Sigmoid()
            )
        elif mode in ('U-MultiClass',):
            self.densenet121.classifier = None
            self.densenet121.Linear_0 = nn.Linear(num_ftrs, out_size)
            self.densenet121.Linear_1 = nn.Linear(num_ftrs, out_size)
            self.densenet121.Linear_u = nn.Linear(num_ftrs, out_size)

        self.mode = mode

        # Official init from torch repo.
        for m in self.densenet121.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self.drop_rate = drop_rate
        self.drop_layer = nn.Dropout(p=drop_rate)

    def forward(self, x):
        features = self.densenet121.features(x)
        out = F.relu(features, inplace=True)

        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)

        if self.drop_rate > 0:
            out = self.drop_layer(out)
        self.activations = out
        if self.mode in ('U-Ones', 'U-Zeros'):
            out = self.densenet121.classifier(out)
        elif self.mode in ('U-MultiClass',):
            n_batch = x.size(0)
            out_0 = self.densenet121.Linear_0(out).view(n_batch, 1, -1)
            out_1 = self.densenet121.Linear_1(out).view(n_batch, 1, -1)
            out_u = self.densenet121.Linear_u(out).view(n_batch, 1, -1)
            out = torch.cat((out_0, out_1, out_u), dim=1)

        return self.activations, out


# end of model defination


def create_model(ema=False, device=None):
    # Network definition
    net = DenseNet121(out_size=N_CLASSES_TOOTH, mode='U-Ones', drop_rate=0.2)
    # model = net.cuda()
    model = net.to(device)
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


batch_size = 16
base_lr = 1e-4
# number of labeled data per batch
labeled_bs = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model(device=device)
ema_model = create_model(ema=True, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.999), weight_decay=5e-4)

# dataset
normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])

train_dataset = Tooth_Dataset(transform=TransformTwice(transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),  # 随机进行旋转
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])))
val_dataset = Tooth_Dataset(transform=TransformTwice(transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),  # 随机进行旋转
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])))
test_dataset = Tooth_Dataset(transform=TransformTwice(transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),  # 随机进行旋转
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])))

# 已标注数据的索引
labeled_idxs = list(range(2726))
# 未标注数据在dataset的索引
unlabeled_idxs = list(range(2726, 22499))
# 对有标签数据和无标签数据的分别采样和集成batch
batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)


# num_worker代表线程数目，每个worker初始化执行函数
def worker_init_fn(worker_id):
    random.seed(1337 + worker_id)


train_dataloader = DataLoader(dataset=train_dataset, batch_sampler=batch_sampler,
                              num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

model.train()
# end of dataload

# defination of loss
import torch
import torch.nn
from torch.nn import functional as F
import numpy as np

"""
The different uncertainty methods loss implementation.
Including:
    Ignore, Zeros, Ones, SelfTrained, MultiClass
"""

METHODS = ['U-Ignore', 'U-Zeros', 'U-Ones', 'U-SelfTrained', 'U-MultiClass']
# CLASS_NUM = [1113, 6705, 514, 327, 1099, 115, 142]
# CLASS_WEIGHT = torch.Tensor([10000 / i for i in CLASS_NUM]).cuda()


class Loss_Zeros(object):
    """
    map all uncertainty values to 0
    """

    def __init__(self):
        self.base_loss = torch.nn.BCELoss(reduction='mean')

    def __call__(self, output, target):
        target[target == -1] = 0
        return self.base_loss(output, target)


class Loss_Ones(object):
    """
    map all uncertainty values to 1
    """

    def __init__(self):
        self.base_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

    def __call__(self, output, target):
        target[target == -1] = 1
        return self.base_loss(output, target)


class cross_entropy_loss(object):
    """
    map all uncertainty values to a unique value "2"
    """

    def __init__(self):
        self.base_loss = torch.nn.CrossEntropyLoss(weight=None, reduction='mean')

    def __call__(self, output, target):
        # target[target == -1] = 2
        output_softmax = F.softmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        return self.base_loss(output_softmax, target.long())


# class weighted_cross_entropy_loss(object):
#     """
#     map all uncertainty values to a unique value "2"
#     """

#     def __init__(self):
#         self.base_loss = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHT, reduction='mean')

#     def __call__(self, output, target):
#         # target[target == -1] = 2
#         output_softmax = F.softmax(output, dim=1)
#         target = torch.argmax(target, dim=1)
#         return self.base_loss(output_softmax, target.long())

def get_UncertaintyLoss(method):
    assert method in METHODS

    if method == 'U-Zeros':
        return Loss_Zeros()

    if method == 'U-Ones':
        return Loss_Ones()


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def entropy_loss(p, C=2):
    ## p N*C*W*H*D
    y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1) / torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent


def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True) / torch.tensor(np.log(C)).cuda()
    return ent


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax - target_softmax) ** 2
    return mse_loss


def cam_attention_map(activations, channel_weight):
    # activations 48*49*1024
    # channel_weight 48*1024
    attention = activations.permute(1, 0, 2).mul(channel_weight)
    attention = attention.permute(1, 0, 2)
    attention = torch.sum(attention, -1)
    attention = torch.reshape(attention, (48, 7, 7))

    return attention


def cam_activation(batch_feature, channel_weight):
    # batch_feature = batch_feature.permute(0,2,3,1)#48 7 7 1024
    # activations = torch.reshape(batch_feature, (batch_feature.shape[0], -1, batch_feature.shape[3]))#48*49*1024

    # attention = activations.permute(1,0,2)#.mul(channel_weight)#49*48*1024
    # attention = attention.permute(1,2,0)#48*1024*49
    # attention = F.softmax(attention, -1)#48*1024*49

    # activations2 = activations.permute(0, 2, 1) #48 1024 49
    # activations2 = activations2 * attention
    # activations2 = torch.sum(activations2, -1)#48*1024
    batch_feature = batch_feature.permute(0, 2, 3, 1)
    # 48*49*1024
    activations = torch.reshape(batch_feature, (batch_feature.shape[0], -1, batch_feature.shape[3]))

    # 49*48*1024
    attention = activations.permute(1, 0, 2).mul(channel_weight)
    # 48*49*1024
    attention = attention.permute(1, 0, 2)
    # 48*49
    attention = torch.sum(attention, -1)
    attention = F.softmax(attention, -1)

    activations2 = activations.permute(2, 0, 1)  # 1024*48*49
    activations2 = activations2 * attention
    activations2 = torch.sum(activations2, -1)  # 1024*48
    # 48 1024
    activations2 = activations2.permute(1, 0)

    return activations2


def relation_mse_loss_cam(activations, ema_activations, model, label):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    weight = model.module.densenet121.classifier[0].weight
    # 48*1024
    channel_weight = label.mm(weight)

    activations = cam_activation(activations.clone(), channel_weight)
    ema_activations = cam_activation(ema_activations.clone(), channel_weight)

    assert activations.size() == ema_activations.size()

    activations = torch.reshape(activations, (activations.shape[0], -1))
    ema_activations = torch.reshape(ema_activations, (ema_activations.shape[0], -1))

    similarity = activations.mm(activations.t())
    norm = torch.reshape(torch.norm(similarity, 2, 1), (-1, 1))
    norm_similarity = similarity / norm

    ema_similarity = ema_activations.mm(ema_activations.t())
    ema_norm = torch.reshape(torch.norm(ema_similarity, 2, 1), (-1, 1))
    ema_norm_similarity = ema_similarity / ema_norm

    similarity_mse_loss = (norm_similarity - ema_norm_similarity) ** 2
    return similarity_mse_loss


def relation_mse_loss(activations, ema_activations):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """

    assert activations.size() == ema_activations.size()

    activations = torch.reshape(activations, (activations.shape[0], -1))
    ema_activations = torch.reshape(ema_activations, (ema_activations.shape[0], -1))

    similarity = activations.mm(activations.t())
    norm = torch.reshape(torch.norm(similarity, 2, 1), (-1, 1))
    norm_similarity = similarity / norm

    ema_similarity = ema_activations.mm(ema_activations.t())
    ema_norm = torch.reshape(torch.norm(ema_similarity, 2, 1), (-1, 1))
    ema_norm_similarity = ema_similarity / ema_norm

    similarity_mse_loss = (norm_similarity - ema_norm_similarity) ** 2
    return similarity_mse_loss


def feature_mse_loss(activations, ema_activations):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """

    assert activations.size() == ema_activations.size()

    activations = torch.reshape(activations, (activations.shape[0], -1))
    ema_activations = torch.reshape(ema_activations, (ema_activations.shape[0], -1))

    # similarity = activations.mm(activations.t())
    # norm = torch.reshape(torch.norm(similarity, 2, 1), (-1, 1))
    # norm_similarity = similarity / norm

    # ema_similarity = ema_activations.mm(ema_activations.t())
    # ema_norm = torch.reshape(torch.norm(ema_similarity, 2, 1), (-1, 1))
    # ema_norm_similarity = ema_similarity / ema_norm

    similarity_mse_loss = (activations - ema_activations) ** 2
    return similarity_mse_loss


def sigmoid_mse_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_softmax = torch.sigmoid(input_logits)
    target_softmax = torch.sigmoid(target_logits)

    loss_fn = torch.nn.MSELoss(reduction='mean')
    mse_loss = loss_fn(input_softmax, target_softmax)
    return mse_loss


def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2) ** 2)


# end of loss defination


# 设置分类loss函数为交叉损失熵
loss_fn = cross_entropy_loss()
# 超参数
iter_num = 0
lr_ = base_lr
ema_decay = 0.99
model.train()

# train
for epoch in range(0, 100):
    time1 = time.time()
    iter_max = len(train_dataloader)
    train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
    # label_batch指的是标签的batch，前面的()内部是image的batch
    for i, (_, _, (image_batch, ema_image_batch), label_batch) in enumerate(train_dataloader):
        time2 = time.time()
        # print('fetch data cost {}'.format(time2-time1))
        image_batch, ema_image_batch, label_batch = image_batch.to(device), ema_image_batch.to(
            device), label_batch.to(device)
        # unlabeled_image_batch = ema_image_batch[labeled_bs:]

        # noise1 = torch.clamp(torch.randn_like(image_batch) * 0.1, -0.1, 0.1)
        # noise2 = torch.clamp(torch.randn_like(ema_image_batch) * 0.1, -0.1, 0.1)
        ema_inputs = ema_image_batch  # + noise2
        inputs = image_batch  # + noise1

        activations, outputs = model(inputs)
        with torch.no_grad():
            ema_activations, ema_output = ema_model(ema_inputs)

        ## calculate the loss
        loss_classification = loss_fn(outputs[:labeled_bs], label_batch[:labeled_bs])
        loss = loss_classification

        ## MT loss (have no effect in the beginneing)
        consistency_weight = get_current_consistency_weight(epoch)
        consistency_dist = torch.sum(
            softmax_mse_loss(outputs, ema_output)) / batch_size  # / dataset.N_CLASSES
        consistency_loss = consistency_weight * consistency_dist

        # consistency_relation_dist = torch.sum(losses.relation_mse_loss_cam(activations, ema_activations, model, label_batch)) / batch_size
        consistency_relation_dist = torch.sum(relation_mse_loss(activations, ema_activations)) / batch_size
        consistency_relation_loss = consistency_weight * consistency_relation_dist * 1

        if epoch > 20:
            loss = loss_classification + consistency_loss + consistency_relation_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        update_ema_variables(model, ema_model, ema_decay, iter_num)
        with torch.no_grad():
            train_l_sum += loss.cpu().item()
            train_acc_sum += (outputs[:labeled_bs].argmax(dim=1) == label_batch[:labeled_bs].argmax(dim=1)).sum().cpu().item()
            n += label_batch[:labeled_bs].shape[0]
            batch_count += 1
        iter_num = iter_num + 1
#     test_acc = evaluate_accuracy(test_dataloader, model, device)
    print('epoch %d, loss %.4f, train acc %.5f, time %.1f sec' % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, time.time() - start))
torch.save(model.state_dict(), 'model.pth')