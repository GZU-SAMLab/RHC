import torch
from torchvision import transforms
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil
from dataset.dataset import get_data
from model.model import get_model


def save_checkpoint(state, is_best, model_name, dataset_name,rate):
    filename = "premodel/"
    cute = model_name + "_" + dataset_name+"_"+str(rate)+".pth"
    torch.save(state, filename + "normal/" + cute)
    if is_best and rate == 0:
        torch.save(state, filename + "best/" + cute)


def _accuracy(net, test_loader):
    net.eval()
    num_total = 0
    num_acc = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.cuda()
            labels = labels.cuda()
            output = net(imgs)
            _, pred = torch.max(output, 1)
            num_acc += torch.sum(pred == labels.detach_())
            num_total += labels.size(0)
    LV = num_acc.detach().cpu().numpy() * 100 / num_total
    return LV


def _top5_accuracy(net, test_loader):
    net.eval()
    num_total = 0
    num_top5_acc = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.cuda()
            labels = labels.cuda()
            output = net(imgs)
            _, top5_pred = torch.topk(output, 5, dim=1)
            num_top5_acc += (top5_pred == labels.view(-1, 1)).sum()
            num_total += labels.size(0)
    LV = num_top5_acc.detach().cpu().numpy() * 100 / num_total
    return LV



def Ban_if(ban_, ban_list):
    length = len(ban_)
    for i in ban_list:
        o = 0
        for j in range(len(i)):
            if i[j] == "*" or i[j] == str(ban_[j]):
                o += 1
        if o == length:
            return True
    return False


model_list = ["Resnet34"]  # "Resnet34","EfficientNet","MobileNet","ResNeXt50","ShuffleNet","SwinTransformer","WideResnet50"
dataset_list = ["CUB"]  # "CUB"
mask_list = [0,0.35,0.45,0.9]  # 0,0.05,0.15,0.25,0.45,0.90

ban_list = []  # "01*"
number_test = 1
epochs = 120

batch_sizes = [100,100,100,100,100]  # 32,32,
GPUS = int(torch.cuda.device_count())
base_lr = 0.0001 * GPUS  # 0.01, 0.001
momentum = 0.9
weight_decay = 1e-4
img_size = 224
step_size = 30
gamma = 0.1

path = 'dataset/'  
save_path = "save/" 

for model_name in model_list:
    for dataset_name in dataset_list:
        train_loader, test_loader = get_data(Path=path, img_size=img_size,
                                             batch_size=GPUS * batch_sizes[model_list.index(model_name)],
                                             dataset_name=dataset_name)
        for number in range(number_test):

            for mask_rate in mask_list:
                if mask_rate == 0:
                    if number != 0:
                        continue
                ban_ = [dataset_list.index(dataset_name), model_list.index(model_name), mask_list.index(mask_rate)]
                if Ban_if(ban_, ban_list):
                    print("此处Pass掉", model_name, " And ", dataset_name)
                    continue

                model = get_model(model_name=model_name, rate=mask_rate, dataset_name=dataset_name)
                model = nn.DataParallel(model).cuda()
                print('-' * 25)
                print("model    :   ", model_name)
                print("dataset  :   ", dataset_name)
                print("Repeat number:   ", number + 1, "号")
                print("mask_rate      :   ", mask_rate * 100, "%")
                print("learning rate      :   ", base_lr)
                print("batch_size  :   ", batch_sizes[model_list.index(model_name)])
                print("iterated limit    :   ", step_size)
                print("gamma       :   ", gamma)
                print("epochs      :   ", epochs)
                print("GPU     :   ", GPUS)
                print('-' * 25)
                print('Training process starts:...')
                if GPUS > 1:
                    print('More than one GPU are used...')

                print(f'{"Epoch":<10}{"TrainLoss":<15}{"TrainAcc":<10}{"TestAcc":<10}{"Acctop5":<10}')
                print('-' * 60)

                criterion = nn.CrossEntropyLoss()
                                                
                solver = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
                schedule = torch.optim.lr_scheduler.StepLR(solver, step_size=step_size, gamma=gamma)

                epochss = np.arange(1, epochs + 1)
                test_acc = list()
                train_acc = list()

                best_acc = 0.0
                best5_acc = 0.0
                best_epoch = 0
                best5_epoch = 0
                model.train(True)
                save_checkpoint(state=model, is_best=True, model_name=model_name, dataset_name=dataset_name,rate=mask_rate)

                for epoch in range(epochs):
                    num_correct = 0
                    train_loss_epoch = list()
                    num_total = 0


                    for imgs, labels in train_loader:
                        solver.zero_grad()
                        imgs = imgs.cuda()
                        labels = labels.cuda()

                        output = model(imgs)
                        loss = criterion(output, labels)
                        _, pred = torch.max(output, 1)
                        num_correct += torch.sum(pred == labels.detach_())
                        num_total += labels.size(0)
                        train_loss_epoch.append(loss.item())
                        loss.backward()
                        solver.step()

                    train_acc_epoch = num_correct.detach().cpu().numpy() * 100 / num_total
                    avg_train_loss_epoch = sum(train_loss_epoch) / len(train_loss_epoch)
                    test_acc_epoch = _accuracy(model, test_loader)
                    test5_acc_epoch = _top5_accuracy(model, test_loader)
                    test_acc.append(test_acc_epoch)
                    train_acc.append(train_acc_epoch)

                    schedule.step()
                    print('{}\t{:.4f}\t{:.2f}%\t{:.2f}%\t{:.2f}%'.format(epoch + 1, avg_train_loss_epoch, train_acc_epoch,
                                                                test_acc_epoch,test5_acc_epoch))

                    is_best = test_acc_epoch >= best_acc
                    is5_best = test5_acc_epoch >= best5_acc
                    best_acc = max(test_acc_epoch, best_acc)
                    best5_acc = max(test5_acc_epoch, best5_acc)
                    save_checkpoint(state=model, is_best=is_best, model_name=model_name, dataset_name=dataset_name, rate=mask_rate)

                    if is_best:
                        best_epoch = epoch
                        print("Best!")

                print("Finished!")
                print("the best epoch:", best_epoch + 1)
                print('(mask={})the best acc:{:.2f}%'.format(mask_rate, best_acc))
                print('(mask={})the best top5 acc:{:.2f}%'.format(mask_rate, best5_acc))

print("Finish It!!!")
