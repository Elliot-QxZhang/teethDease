import os
import random
import time
import math
import torch
from torch.optim import lr_scheduler
from torch.utils.data import Dataset

from dataset import K_folder_dataset
from test_model import evaluate_accuracy


def chunks(arr, m):
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]


def get_k_fold_data(imgs, k, i):
    list_of_list = chunks(arr=imgs, m=k)
    valid_list = list_of_list[i]
    train_list = []
    for j in range(0, len(list_of_list)):
        if j == i:
            continue
        train_list += list_of_list[j]
    return train_list, valid_list


def train(train_iter, test_iter, model, device, i=1):
    loss = torch.nn.CrossEntropyLoss()
    lr, num_epochs = 0.001, 100
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0003)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=0, last_epoch=-1)

    print('train begin with ' + str(i))
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for img, label in train_iter:
            X = img.to(device)
            y = label.to(device)
            y_hat = model(X)
            l = loss(y_hat, y.long())
            optimizer.zero_grad()  # 每一次迭代完成之前都要清楚梯度，每一次迭代的梯度不能累加
            l.backward()
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                train_l_sum += l.cpu().item()
                train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
                n += y.shape[0]
                batch_count += 1
                test_acc = evaluate_accuracy(test_iter, model)
        print('epoch %d, loss %.4f, train acc %.5f, test acc %.5f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
    torch.save(model.state_dict(), 'model-' + str(i) + '.pth')
    print('acc:' + str(evaluate_accuracy(test_iter, model)))


def k_folder_train(k, device, model):
    imgs = []
    test_imgs = []
    for dirname, _, filenames in os.walk('../input/tooth-project-3/k-folder/k-folder/caries'):
        for filename in filenames:
            imgs.append((str(os.path.join(dirname, filename)), 1))
    for dirname, _, filenames in os.walk('../input/tooth-project-3/k-folder/k-folder/health'):
        for filename in filenames:
            imgs.append((str(os.path.join(dirname, filename)), 0))
    for dirname, _, filenames in os.walk('../input/tooth-project-3/k-folder/k-folder/testcaries'):
        for filename in filenames:
            test_imgs.append((str(os.path.join(dirname, filename)), 1))
    for dirname, _, filenames in os.walk('../input/tooth-project-3/k-folder/k-folder/testhealth'):
        for filename in filenames:
            test_imgs.append((str(os.path.join(dirname, filename)), 0))
    random.shuffle(imgs)
    for i in range(0, k):
        train_list, valid_list = get_k_fold_data(imgs, k, i)
        train_set = K_folder_dataset(train_list)
        test_set = K_folder_dataset(valid_list)
        batch_size = 32
        train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
        train(train_iter, test_iter, model, device)
