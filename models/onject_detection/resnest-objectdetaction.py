#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pip install fvcore


# # **Dataset & Dataloader**

# In[ ]:


# 引入这个cell中的包
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import numpy as np
import cv2

# 根据提供的记录生成数据集
class TeethDataset(Dataset):
    def __init__(self, 
                 dataframe, 
                 classes,
                 images_dir= "/kaggle/input/teeth-test/images/images/"):
        #dataframe： 本数据集使用的标签记录，使用pandas包构造操作
        #images_dir:  图片所在的路径
        super().__init__()
        self.images_dir = images_dir
        self.image_names = dataframe["name"].unique()
        self.dataframe = dataframe
        self.classes_transform =  {classes[i] : i for i in range(len(classes))}
        # 图片格式转换
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        image_name = self.image_names[index]
        records = self.dataframe[self.dataframe["name"] == image_name]
        
        # 使用opencv包 打开图片
        image = cv2.imread(self.images_dir + image_name, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        image = self.transform(image)

        #Get bounding box co-ordinates for each box
        boxes = records[['x1', 'y1', 'x2', 'y2']].values
        #Getting labels for each box
        temp_labels = records[['classname']].values
        labels = []
        for label in temp_labels:
            label = class_to_int[label[0]]
            labels.append(label)
        # 转换成为相应的torch.Tensor格式
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        #Creating target
        target = {'boxes': boxes, 'labels': labels}
#         target['boxes'] = boxes
#         target['labels'] = labels
        
        return image, target, image_name
        
        
def collate_fn(batch):
    return tuple(zip(*batch))


def teeth_dataloader(classes, 
                     stage = 0,
                     folder = 5,
                     batch_size= 6, 
                     collate_fn= collate_fn, 
                     images_dir = "/kaggle/input/teeth-test/images/images/"):
    if folder is None:
        k_folder = False
    else:
        k_folder = True
    if k_folder and stage >= folder:
        stage = folder-1
    
    # dataframe是全局变量
    image_names = dataframe["name"].unique()
    images_count = len(image_names)
    
    if k_folder and stage == folder-1:
        test_image_names = image_names[int(stage/folder*images_count):]
    elif k_folder:
        test_image_names =         image_names[int(stage/folder*images_count):int((stage+1)/folder*images_count)]
    else:
        test_image_names = image_names[int(stage*images_count):]
    test_dataframe = dataframe[[name in test_image_names for name in dataframe["name"]]]
    train_dataframe = pd.concat([dataframe, test_dataframe]).drop_duplicates(keep=False)
    print(len(train_dataframe), len(test_dataframe))
    
    train_dataset = TeethDataset(train_dataframe, classes, images_dir)
    test_dataset = TeethDataset(test_dataframe, classes, images_dir)


    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                             shuffle=True, num_workers=4, collate_fn = collate_fn)

    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                           shuffle=False, num_workers=4, collate_fn = collate_fn)
    return train_iter, test_iter

# # 测试
# import pandas as pd
# classes = ['background', 'tooth-r']
# class_to_int = {classes[i] : i for i in range(len(classes))}
# int_to_class = {i : classes[i] for i in range(len(classes))}

# input_dir = "/kaggle/input/teeth-test/"
# imgs_dir = input_dir+"images/images/"
# train_dataframe = pd.read_csv(input_dir + "train.csv")
# test_dataframe = pd.read_csv(input_dir + "eval.csv")
# dataframe = pd.concat([train_dataframe,test_dataframe],axis=0)
#     ## 打乱记录顺序
# dataframe.sample(frac=1)
# train_iter, test_iter = teeth_dataloader(classes= classes, stage=2, folder=3)
# for images, targets, _ in train_iter:
#     #Loading images & targets on device
#     images = list(image for image in images)
#     targets = [{k: v for k, v in t.items()} for t in targets]
#     print(len(images))
# for images, targets, image_name in test_iter:
#     print(len(images))


# # **Model Section**

# * **attention**
# 

# *  **Backbone preparation**

# In[ ]:


# pakages:
# torchvision == 0.8.1+
from torchvision.models.detection.backbone_utils import BackboneWithFPN, resnet_fpn_backbone
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.models import vgg, resnet
from torchvision.ops import misc as misc_nn_ops
import torch

def vgg_fpn_backbone(backbone_name,
                    pretrained = False,
                    returned_layers=None):    
    backbone = vgg.__dict__[backbone_name](pretrained=pretrained)
    # 去掉最后的池化层：
    backbone.features = backbone.features[:-1]
    
    if returned_layers is None:
        returned_layers = [0]
    return_layers = {f'features': str(v) for v, k in enumerate(returned_layers)}
    # 这里是给FPN的返回层，可以查看github resnet 代码，其中有layer1... 而vgg只有features，源代码没有明显分层
    
    in_channels_list = [512]
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)


def resnest_fpn_backbone(backbone_name,
                            pretrained,
                            trainable_layers=4,
                            returned_layers=None,
                            extra_blocks=None):
    
    # get list of models
    torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)

    # load pretrained models, using ResNeSt-50 as an example
    backbone = torch.hub.load('zhanghang1989/ResNeSt', backbone_name, pretrained=pretrained, 
                             num_classes = 2)

    # select layers that wont be frozen
    assert 0 <= trainable_layers <= 5
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    if trainable_layers == 5:
        layers_to_train.append('bn1')
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    assert min(returned_layers) > 0 and max(returned_layers) < 5
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)

def get_backbone(name, pretrained=False, trainable_layers=3, returned_layers=None):
    # returned_layers 影响anchors_generator的anchors_size个数！
    # name: vgg,16; resnet,50
    name = name.split(',')
    backbone_name = name[0]+name[1]
    if name[0] == 'vgg':
        backbone = vgg_fpn_backbone(backbone_name, pretrained, returned_layers)
        return backbone_name, backbone
    elif name[0] == 'resnet':
        backbone = resnet_fpn_backbone(backbone_name, pretrained= False)
        return backbone_name, backbone
    elif name[0] == 'resnest':
        backbone = resnest_fpn_backbone(backbone_name, pretrained=False)
        return backbone_name, backbone
    else:
        # 这里需要补充，暂时这样没有问题
        return name, None
# 测试
# backbone = vgg_fpn_backbone('vgg16')
# backbone = resnet_fpn_backbone('resnet50')
# backbone_name, backbone = get_backbone('resnet,50', False)
# print(backbone)


# *  **Model with backbone**

# In[ ]:


from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
# backbone为元组， [0]:name, [1]: network
def get_faster_rcnn_model(backbone, num_classes):
#     anchors_generator 的尺度很重要，对于resnet，代码中会自动给，但是vgg需要自己指定！ 
#     features = backbone(image)
#     for key in features:
#         print(key,"的大小：",features[key].shape)
    name = backbone[0]
    if name[:3] == 'vgg':
        # 如果想要更充分利用FPN，需要考虑重新实现以下VGG，使其各层特征图可以展现出来
        anchor_generator = AnchorGenerator(sizes=((32), (64)),
                                   aspect_ratios=((0.5, 1.0, 2.0),)*2)
        model = FasterRCNN(backbone[1],
                           num_classes=num_classes,
                           rpn_anchor_generator=anchor_generator)
    elif name[:6] == 'resnet':
        # anchor_sizes与return_layer相关，为：len(anchor_sizes) = len(return_layer) + pool(extra_block)
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        model = FasterRCNN(backbone[1],
                           num_classes=num_classes,
                           rpn_anchor_generator=anchor_generator)
    elif name[:7] == 'resnest':
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        model = FasterRCNN(backbone[1],
                           num_classes=num_classes,
                           rpn_anchor_generator=anchor_generator)
    else:
        # 暂不指定，不会出错
        model = None
    return model
# 测试
# backbone_name, backbone = get_backbone('resnet,50', False)
# model = get_faster_rcnn_model((backbone_name, backbone), 2)
# print(model)


# # **Plain_train & Epoch_train**

# In[ ]:


import time
import torch
import numpy as np
from torch import optim

def analyze_model_out(model_out, epoch):
    temp_model_out = {'epoch': epoch}
    tmp = model_out[0]
    keys = [key for key, _ in tmp.items()]
    for k in keys:
        temp_model_out[k] = np.mean([model_out_iter[k].item() for model_out_iter in model_out])
    model_out = temp_model_out
    return model_out

def epoch_train(device, train_iter, epoch):
    model.train(mode=True)
    model_out = []
    for images, targets, _ in train_iter:
        #Loading images & targets on device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        #Forward propagation
        model_out_iter = model(images, targets)
        losses_sum = sum(loss for loss in model_out_iter.values())
        
        #Reseting Gradients
        optimizer.zero_grad()
        
        #Back propagation
        losses_sum.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        model_out.append(model_out_iter)
            
    lr_scheduler.step()   
    return analyze_model_out(model_out, epoch)

def analyze_print(epoch, model_out, use_time, epoch_test, teeth, mean_AP):
    losses_sum = sum(loss for loss in model_out.values())
    losses_sum -= model_out['epoch']
    
    print(f'Epoch{epoch} sum train loss is {losses_sum}')
    if epoch_test:
        print('teeth_AP: ',teeth[-1], ' mean_AP: ', mean_AP)
        print('teeth_rec: ', teeth[0][-1], ' teeth_prec: ', teeth[1][-1])
    print('time:', use_time)

def train_core(epoch_start_end, train_iter, test_msg):
    # test: [0]: epoch_test? bool; [1]: test_dataframe
    epoch_test, test_iter = test_msg
#     epoch_start_end: [0]: epoch_start; [1]: epoch_end
    epoch_start, epoch_end = epoch_start_end
    
#     print(type(epoch_test), type(test_iter))
    epoch = epoch_start
    model_outs = []
    while epoch < epoch_end:
        start_time = time.time()
        
        model_out = epoch_train(device, train_iter, epoch)
        model_outs.append(model_out)
        if epoch_test:
            teeth, mean_AP = test(test_iter, device)
        
        cost = time.time() - start_time
        analyze_print(epoch, model_out, cost, epoch_test, teeth, mean_AP)
        epoch += 1
    return model_outs

# # 测试
# model_outs = []
# for i in range(5):
#     model_outs.append({'cls': i, 'breg': i, 'obj': i, 'rpn': i})
# print(analyze_model_out(model_outs))


# # **Test & PR-AP**

# In[ ]:


import torch
import numpy as np

def prec_rec(outs, img_names, targets, classname):
    ovthresh = 0.5
#     重新整合targets
    targets_d = []
    for target in targets:
        target_tmp = [{'box': box, 'label': label} for box, label in zip(target['boxes'], target['labels'])]
        targets_d.append(target_tmp)
        
    targets = targets_d
    
    recs = {}
    for img, target in zip(img_names, targets):
        recs[img] = target
    tooth_prec, tooth_rec, wisdom_prec, wisdom_rec = 0, 0, 0, 0
    class_recs = {}
    npos = 0
    for img in img_names:
        R = [obj for obj in recs[img] if obj['label'] == classname]
        bbox = np.array([x['box'].numpy() for x in R])
        # difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + len(R)
        class_recs[img] = {'bbox': bbox,
                           # 'difficult': difficult,
                           'det': det}

    image_ids = []
    confidence = []
    BB = []
    old_len = 0
    for img, out_i in zip(img_names, outs):
        out_i['scores'].tolist()
        out_i['labels'].tolist()
        confidence += [score for score, label in zip(out_i['scores'], out_i['labels']) if label == classname]
        BB += [torch.Tensor.cpu(box).numpy() for box, label in zip(out_i['boxes'], out_i['labels']) if label == classname]
        image_ids += [img]*(len(confidence) - old_len)
        old_len = len(confidence)

    confidence = np.array(confidence)
    BB = np.array(BB)

    nd = len(confidence)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    if BB.shape[0] > 0:
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                    (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                    (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            # if not R['difficult'][jmax]:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.
    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    return rec, prec


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:  # 使用07年方法
        # 11 个点
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])  # 插值
            ap = ap + p / 11.
    else:  # 新方式，计算所有点
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision 曲线值（也用了插值）
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def test(test_iter, device):
    with torch.no_grad():
        model.eval()
        pred = []
        img_names = []
        tars = []
        for images, targets, image_names in test_iter:
            #Loading images & targets on device
            images = list(image.to(device) for image in images)
            out = model(images)
            pred.extend(out)
            img_names.extend(image_names)
            tars.extend(targets)
        
        teeth_rec, teeth_prec = prec_rec(pred, img_names, tars, 1)
        teeth_AP = voc_ap(teeth_rec, teeth_prec, use_07_metric=False)
        teeth = (teeth_rec, teeth_prec, teeth_AP)
        meanAP = -1
#         wisdom_rec, winsdom_prec = prec_rec(pred_out, img_names, tars, 1)
#         if ('wisdom_rec' in locals().keys()) and ('wisdom_prec' in locals().keys()):
#             wisdom_ap = voc_ap(wisdom_rec, wisdom_prec)
#             meanAP = float(teeth_ap + wisdom_ap)/2
#         else:
        meanAP = teeth_AP
        
        return teeth, meanAP


# # **Get dataframe**

# In[ ]:


import pandas as pd

# the data preparation
    # path
input_dir = "/kaggle/input/teeth-test/"
imgs_dir = input_dir+"images/images/"
dataframe = pd.read_csv(input_dir + "dataframe.csv")
# train_dataframe = pd.read_csv(input_dir + "train.csv")
# test_dataframe = pd.read_csv(input_dir + "eval.csv")
# dataframe = pd.concat([train_dataframe,test_dataframe],axis=0).sample(frac=1).reset_index(drop=True)
# #     ## 打乱记录顺序
# print(len(dataframe))
# print(dataframe)
# dataframe.to_csv('/kaggle/working/dataframe.csv', index=0)


# # **Model, optimizer Code**

# In[ ]:


import torch
from torch import optim

# classes
classes = ['background', 'tooth-r']
class_to_int = {classes[i] : i for i in range(len(classes))}
int_to_class = {i : classes[i] for i in range(len(classes))}
num_classes = len(classes)

# Model section
name = "resnest,50"
backbone_name, backbone = get_backbone(name, pretrained=False, trainable_layers=3, returned_layers=None)
model = get_faster_rcnn_model((backbone_name, backbone), num_classes)
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

# checkpoint = torch.load('/kaggle/input/teeth-test/vgg16_checkout.pth')
# model.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer'])
#lr_scheduler.load_state_dict(checkpoint['lr'])

# GPU是否可用
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()
model.to(device)

# optimizer
lr = 0.001
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(params, lr = lr)

# learn_scheduler
epoch_count = 80
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch_count, eta_min=0,                                                     last_epoch=-1, verbose=False)

# lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 40, gamma=0.1)
#epoch
epoch = (0, epoch_count)

train_iter, test_iter = teeth_dataloader(classes= classes, stage=0.7, folder=None, batch_size= 4)
model_outs = train_core(epoch, train_iter, (True, test_iter))
print(model_outs)


# # **Save & Download**

# In[ ]:


import os

PATH = '/kaggle/working/checkpoint.pth'
torch.save({'model_state_dict': model.state_dict(),
            'model_outs': model_outs,
            'optimizer': optimizer,
            'lr': lr_scheduler
            }, PATH)

os.chdir('/kaggle/working')
print(os.getcwd())
print(os.listdir("/kaggle/working"))
from IPython.display import FileLink
FileLink('checkpoint.pth')


# In[ ]:


model.to(torch.device('cpu'))
import matplotlib.pyplot as plt
def plot_img(image_name, df):
    
    fig, ax = plt.subplots(2, 1, figsize = (14, 14))
    ax = ax.flatten()
    
    bbox = df[df['name'] == image_name]
    img_path = os.path.join(imgs_dir, image_name)
    
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    image3 = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])(image).unsqueeze(0)
    model.eval()
    pred_out = model(image3)
    pred_bbox = [box for box in pred_out[0]['boxes']]
    pred_scores = [score for score in pred_out[0]['scores']]
    image2 = image
    
    
    for idx, row in bbox.iterrows():
        x1 = row['x1']
        y1 = row['y1']
        x2 = row['x2']
        y2 = row['y2']
        label = row['classname']
        
#         cv2.rectangle(image2, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 3)
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         cv2.putText(image2, label, (int(x1),int(y1-10)), font, 1, (255,0,0), 2)
    
    ax[0].set_title('Image with Bondary Box')
    ax[0].imshow(image2)
    
    for (x1, y1, x2, y2), score in zip(pred_bbox, pred_scores):
        cv2.rectangle(image, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = str(score.data)
        cv2.putText(image, text[7: -1], (int(x1),int(y1-10)), font, 1, (255,0,0), 2)
    ax[1].set_title('Pred Image')
    ax[1].imshow(image)

    plt.show()
    print(len(pred_scores))

plot_img("000575.TIF", dataframe)

