import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from torch.utils.data import Dataset


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():  # 在计算精度的过程当中不累加梯度
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()  # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()  # 改回训练模式
            else:
                if ('is_training' in net.__code__.co_varnames):  # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


def pred(test_set, model, device):
    y_pred = []
    y_true = []
    TP, FP, FN, TN = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        batch_size = 1
        test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
        with torch.no_grad():  # 在计算精度的过程当中不累加梯度
            for X, y in test_iter:
                if isinstance(model, torch.nn.Module):
                    model.eval()  # 评估模式, 这会关闭dropout
                    y_true.append(y.cpu().item())
                    # print(model(X.to(device))[0])
                    #                 print(y.cpu().item())
                    y_pred.append((F.softmax(model(X.to(device))[0], dim=0))[0].cpu().item())
                    # y_pred.append()
                    if model(X.to(device)).argmax(dim=1) == y.to(device) and y.cpu().item() == 1:
                        TP += 1
                    elif model(X.to(device)).argmax(dim=1) == y.to(device) and y.cpu().item() != 1:
                        TN += 1
                    elif model(X.to(device)).argmax(dim=1) != y.to(device) and y.cpu().item() != 1:
                        FP += 1
                    elif model(X.to(device)).argmax(dim=1) != y.to(device) and y.cpu().item() == 1:
                        FN += 1
    return y_true, y_pred, TP, FP, FN, TN


def get_recall(test_set, model, device):
    _, _, TP, FP, FN, TN = pred(test_set, model, device)
    if (TP + FN) != 0:
        return TP / (TP + FN)
    else:
        return 0


def get_precision(test_set, model, device):
    _, _, TP, FP, FN, TN = pred(test_set, model, device)
    if (TP + FP) != 0:
        return TP / (TP + FP)
    else:
        return 0


def pr_curve(title, test_set, model, device, path):
    y_true, y_pred, _, _, _, _ = pred(test_set, model, device)
    plt.figure("P-R Curve")
    plt.title(title)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    y_pred_1 = []
    for y in y_pred:
        y_pred_1.append(1 - y)
    precision, recall, thresholds = precision_recall_curve(np.array(y_true), np.array(y_pred_1))
    plt.plot(recall, precision)
    plt.savefig(path, dpi=1200)
    plt.show()


def plot_matrix(test_set, model, device, labels_name, title=None, thresh=0.8, axis_labels=None, path=None):
    y_true, y_pred, _, _, _, _ = pred(test_set, model, device)
    # 利用sklearn中的函数生成混淆矩阵并归一化
    y_pred_1 = []
    for y in y_pred:
        if y >= 0.5:
            y_pred_1.append(0)
        else:
            y_pred_1.append(1)
    cm = metrics.confusion_matrix(y_true, y_pred_1, labels=labels_name, sample_weight=None)  # 生成混淆矩阵
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化

    # 画图，如果希望改变颜色风格，可以改变此部分的cmap=pl.get_cmap('Blues')处
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.colorbar()  # 绘制图例

    # 图像标题
    if title is not None:
        plt.title(title)
    # 绘制坐标
    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = labels_name
    plt.xticks(num_local, axis_labels, rotation=45)  # 将标签印在x轴坐标上， 并倾斜45度
    plt.yticks(num_local, axis_labels)  # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if int(cm[i][j] * 100 + 0.5) > 0:
                plt.text(j, i, format(int(cm[i][j] * 100 + 0.5), 'd') + '%',
                         ha="center", va="center",
                         color="white" if cm[i][j] > thresh else "black")  # 如果要更改颜色风格，需要同时更改此行
    # 显示
    plt.savefig(path)
    plt.show()
