import torch
import torch.nn as nn
from torchvision import models
import os
import numpy as np
from torchvision import utils as vutils
from utils.weight_init import weight_init_kaiming
import torch.nn.functional as F
import random
from TransFG import VisionTransformer as vit

model_path = "/home/wcy/testnet/premodel/best/"

        
class Resnet50(nn.Module):
    def __init__(self, n_class):
        super(Resnet50, self).__init__()
        self.n_class = n_class
        self.base_model = models.resnet50(pretrained=True)
        self.base_model.fc = nn.Linear(2048,n_class)
        self.base_model.fc.apply(weight_init_kaiming)
        
    def forward(self, X):
        X = self.base_model(X)
        return X

class Densenet169(nn.Module):
    def __init__(self,n_class):
        super(Densenet169, self).__init__()
        self.n_class = n_class
       
        self.base_model = models.densenet169(pretrained=True)
        self.base_model.classifier = nn.Linear(1664,self.n_class)
        
        self.base_model.classifier.apply(weight_init_kaiming)
        
    def forward(self, X):
        X = self.base_model(X)
        return X

class ViT(nn.Module):
    def __init__(self,n_class):
        super(ViT, self).__init__()
        self.n_class = n_class
        self.base_model = vit(img_size=224, num_classes=n_class, smoothing_value=0.0, zero_head=True)
        self.base_model.load_from(np.load("/home/wcy/testnet/vit.npz"))
        self.base_model.part_head = nn.Linear(768,self.n_class)
        self.base_model.part_head.apply(weight_init_kaiming)
        
    def forward(self, X):
        X = self.base_model(X)
        return X    
    
    
def get_model(model_name, dataset_name):
    n_class = None
    #dataset_list = ["cifar100","VOCdevkit","CUB"]
    #model_list = ["DenseNet169","ViT","Resnet50"]
    model_path = "/home/wcy/testnet/premodel/best/"
    model_path = model_path + model_name+'_'+dataset_name+".pth"
    if dataset_name == "CUB":      
      n_class = 200
    elif dataset_name == "cifar100":
      n_class = 100
    elif dataset_name == "VOCdevkit":
      n_class = 20
    elif dataset_name == "ImageNet-1k":
      n_class = 1000
    else:
      print("没有这个数据集:",dataset_name)
      print(1/0)
    
    if model_name == "DenseNet169":
      return Densenet169(n_class)
    elif model_name == "Resnet50":
      return Resnet50(n_class)
    elif model_name == "ViT":
      return ViT(n_class)
    else:
      print("没有这个模型：",model_name)
      print(1/0)
      

class VGG16(nn.Module):
    def __init__(self,n_class,rate):
        super(VGG16, self).__init__()
        self.n_class = n_class
        self.base_model = models.vgg16(pretrained=True)
        self.base_model.classifier[6] = FC_(rate,4096,self.n_class)#nn.Linear(4096,self.n_class)
        self.base_model.classifier[6].apply(weight_init_kaiming)
        
    def forward(self, X):
        X = self.base_model(X)
        return X

class Alexnet(nn.Module):
    def __init__(self,n_class,rate):
        super(Alexnet, self).__init__()
        self.n_class = n_class
        self.base_model = models.alexnet(pretrained=True)
        self.base_model.classifier[6] = FC_(rate,4096,self.n_class)#nn.Linear(4096,self.n_class)
        self.base_model.classifier[6].apply(weight_init_kaiming)
        
    def forward(self, X):
        X = self.base_model(X)
        return X  