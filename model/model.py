import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights,ResNet34_Weights,ResNet50_Weights,ResNet101_Weights,ResNet152_Weights,EfficientNet_V2_S_Weights,MobileNet_V3_Small_Weights,ResNeXt50_32X4D_Weights,ShuffleNet_V2_X0_5_Weights,SqueezeNet1_0_Weights,Swin_T_Weights,MobileNet_V3_Large_Weights,ShuffleNet_V2_X2_0_Weights,Wide_ResNet50_2_Weights,DenseNet121_Weights,DenseNet161_Weights,DenseNet201_Weights,DenseNet169_Weights
import sys
import os

import numpy as np
from torchvision import utils as vutils
from utils.weight_init import weight_init_kaiming
import torch.nn.functional as F
import random
from TransFG import VisionTransformer as vit

#model_path = "/home/zaowuyu/project/testnet/premodel/best/"
class FC_(nn.Module):
    def __init__(self,rate,inp,outp):
    
        super(FC_,self).__init__()
        self.mask_block = random.sample(range(inp),int(inp*(1-rate)))
        self.mask_block = torch.tensor(self.mask_block)
        self.layer = nn.Linear(int(inp*(1-rate)),outp).cuda()

    def forward(self,x): 
        x = x.reshape([x.shape[0],x.shape[1]])
        x = x[:,self.mask_block]
        x = self.layer(x)
        return x
    
class FC_1(nn.Module):
    def __init__(self,rate,inp,outp,already):
    
        super(FC_1,self).__init__()
        self.mask_block = random.sample(range(inp),int(inp*(rate)))
        self.mask_block = torch.tensor(self.mask_block)
        self.layer = already.layer
        self.layer.weight.data = self.layer.weight.data[:,self.mask_block]

    def forward(self,x): 
        x = x.reshape([x.shape[0],x.shape[1]])
        x = x[:,self.mask_block]
        x = self.layer(x)
        return x


        
class Resnet18(nn.Module):
    def __init__(self, n_class,rate,New,model_path):
        super(Resnet18, self).__init__()
        self.n_class = n_class
        if New:
            self.base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            self.base_model.fc = nn.Linear(512,n_class)
        else:
            self.base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            #self.base_model.fc = nn.Linear(512,n_class)
            self.base_model.fc = FC_(rate = rate, inp = 512, outp = n_class)
        self.base_model.fc.apply(weight_init_kaiming)
        
    def forward(self, X):
        X = self.base_model(X)
        return X

class Densenet169(nn.Module):
    def __init__(self,n_class,rate,New,model_path):
        super(Densenet169, self).__init__()
        self.n_class = n_class
        if New:
            self.base_model = models.densenet169(weights=DenseNet169_Weights.DEFAULT)
            self.base_model.classifier = nn.Linear(self.base_model.classifier.in_features,n_class)
        else:
            self.base_model = torch.load(model_path).module.base_model
            self.base_model.classifier = FC_(rate = rate, inp = self.base_model.classifier.in_features, outp = n_class)
        self.base_model.classifier.apply(weight_init_kaiming)
        
    def forward(self, X):
        X = self.base_model(X)
        return X

class ViT(nn.Module):
    def __init__(self,n_class,rate,New,model_path):
        super(ViT, self).__init__()
        self.n_class = n_class
        if New:            
            self.base_model = vit(img_size=224, num_classes=200, smoothing_value=0.0, zero_head=True)
            self.base_model.load_from(np.load("/home/zaowuyu/project/testnet/vit.npz"))
            self.base_model.part_head = nn.Linear(768,self.n_class)
        else:
            self.base_model = torch.load(model_path).module.base_model
            self.base_model.part_head = FC_(rate,768,self.n_class)#nn.Linear(4096,self.n_class)
        self.base_model.part_head.apply(weight_init_kaiming)
        
    def forward(self, X):
        X = self.base_model(X)
        return X    
    
    
def get_model(model_name, rate, dataset_name):
    n_class = None
    #dataset_list = ["cifar100","VOCdevkit","CUB"]
    #model_list = ["DenseNet169","ViT","Resnet18"]
    model_path = "premodel/test/"
    model_path = model_path + model_name+'_'+dataset_name+"_0.pth"
    print(model_path)
    if dataset_name == "CUB":
      n_class = 200
    elif dataset_name == "cifar100":
      n_class = 100
    elif dataset_name == "VOCdevkit":
      n_class = 20
    elif dataset_name == "mini":
      n_class = 100
    elif dataset_name == "PDI-C":
      n_class = 120
    elif dataset_name == "tiny":
      n_class = 200
    elif dataset_name == "aircraft":
      n_class = 100
    else:
      print("没有这个数据集:",dataset_name)
      print(1/0)
    
    if rate == 0:
        New = True 
    else:
        New = False
    
    if model_name == "DenseNet169":
      return Densenet169(n_class,rate,New,model_path)
    elif model_name == "Resnet18":
      return Resnet18(n_class,rate,New,model_path)
    elif model_name == "ViT":
      return ViT(n_class,rate,New,model_path)
    elif model_name == "Resnet34":
      return Resnet34(n_class,rate,New,model_path)
    elif model_name == "Resnet50":
      return Resnet50(n_class,rate,New,model_path)
    elif model_name == "Resnet101":
      return Resnet101(n_class,rate,New,model_path)
    elif model_name == "Resnet152":
      return Resnet152(n_class,rate,New,model_path)
    elif model_name == "EfficientNet":
      return EfficientNet(n_class,rate,New,model_path)
    elif model_name == "MobileNet":
      return MobileNet(n_class,rate,New,model_path)
    elif model_name == "ResNeXt50":
      return ResNeXt50(n_class,rate,New,model_path)
    elif model_name == "ShuffleNet":
      return ShuffleNet(n_class,rate,New,model_path)
    elif model_name == "SqueezeNet":
      return SqueezeNet(n_class,rate,New,model_path)
    elif model_name == "SwinTransformer":
      return SwinTransformer(n_class,rate,New,model_path)
    elif model_name == "WideResnet50":
      return SwinTransformer(n_class,rate,New,model_path)
    if model_name == "DenseNet121":
      return Densenet121(n_class,rate,New,model_path)
    if model_name == "DenseNet161":
      return Densenet161(n_class,rate,New,model_path)
    if model_name == "DenseNet201":
      return Densenet201(n_class,rate,New,model_path)
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
        
        
class Resnet34(nn.Module):
    def __init__(self, n_class,rate,New,model_path):
        super(Resnet34, self).__init__()
        self.n_class = n_class
        if New:
            self.base_model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
            self.base_model.fc = nn.Linear(self.base_model.fc.in_features,n_class)
        else:
            self.base_model = torch.load(model_path).module.base_model
            self.base_model.fc = FC_(rate = rate, inp = self.base_model.fc.in_features, outp = n_class)
        self.base_model.fc.apply(weight_init_kaiming)
        
    def forward(self, X):
        X = self.base_model(X)
        return X
        
    
class Resnet50(nn.Module):
    def __init__(self, n_class,rate,New,model_path):
        super(Resnet50, self).__init__()
        self.n_class = n_class
        if New:
            self.base_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.base_model.fc = nn.Linear(self.base_model.fc.in_features,n_class)
        else:
            self.base_model = torch.load(model_path).module.base_model
            self.base_model.fc = FC_(rate = rate, inp = self.base_model.fc.in_features, outp = n_class)
        self.base_model.fc.apply(weight_init_kaiming)
        
    def forward(self, X):
        X = self.base_model(X)
        return X
        
        
class Resnet101(nn.Module):
    def __init__(self, n_class,rate,New,model_path):
        super(Resnet101, self).__init__()
        self.n_class = n_class
        if New:
            self.base_model = models.resnet101(weights=ResNet101_Weights.DEFAULT)
            self.base_model.fc = nn.Linear(self.base_model.fc.in_features,n_class)
        else:
            self.base_model = torch.load(model_path).module.base_model
            self.base_model.fc = FC_(rate = rate, inp = self.base_model.fc.in_features, outp = n_class)
        self.base_model.fc.apply(weight_init_kaiming)
        
    def forward(self, X):
        X = self.base_model(X)
        return X
        
        
class Resnet152(nn.Module):
    def __init__(self, n_class,rate,New,model_path):
        super(Resnet152, self).__init__()
        self.n_class = n_class
        if New:
            self.base_model = models.resnet152(weights=ResNet152_Weights.DEFAULT)
            self.base_model.fc = nn.Linear(self.base_model.fc.in_features,n_class)
        else:
            self.base_model = torch.load(model_path).module.base_model
            self.base_model.fc = FC_(rate = rate, inp = self.base_model.fc.in_features, outp = n_class)
        self.base_model.fc.apply(weight_init_kaiming)
        
    def forward(self, X):
        X = self.base_model(X)
        return X
        
        
class EfficientNet(nn.Module):
    def __init__(self, n_class,rate,New,model_path):
        super(EfficientNet, self).__init__()
        self.n_class = n_class
        if New:
            self.base_model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
            self.base_model.classifier[-1] = nn.Linear(self.base_model.classifier[-1].in_features,n_class)
        else:
            self.base_model = torch.load(model_path).module.base_model
            self.base_model.classifier[-1] = FC_(rate = rate, inp = self.base_model.classifier[-1].in_features, outp = n_class)
        self.base_model.classifier[-1].apply(weight_init_kaiming)
        
    def forward(self, X):
        X = self.base_model(X)
        return X
        
        
class MobileNet(nn.Module):
    def __init__(self, n_class,rate,New,model_path):
        super(MobileNet, self).__init__()
        self.n_class = n_class
        if New:
            self.base_model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
            self.base_model.classifier[-1] = nn.Linear(self.base_model.classifier[-1].in_features,n_class)
        else:
            self.base_model = torch.load(model_path).module.base_model
            self.base_model.classifier[-1] = FC_(rate = rate, inp = self.base_model.classifier[-1].in_features, outp = n_class)
        self.base_model.classifier[-1].apply(weight_init_kaiming)
        
    def forward(self, X):
        X = self.base_model(X)
        return X
        
        
class ResNeXt50(nn.Module):
    def __init__(self, n_class,rate,New,model_path):
        super(ResNeXt50, self).__init__()
        self.n_class = n_class
        if New:
            self.base_model = models.resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
            self.base_model.fc = nn.Linear(self.base_model.fc.in_features,n_class)
        else:
            self.base_model = torch.load(model_path).module.base_model
            self.base_model.fc = FC_(rate = rate, inp = self.base_model.fc.in_features, outp = n_class)
        self.base_model.fc.apply(weight_init_kaiming)
        
    def forward(self, X):
        X = self.base_model(X)
        return X
        
        
class ShuffleNet(nn.Module):
    def __init__(self, n_class,rate,New,model_path):
        super(ShuffleNet, self).__init__()
        self.n_class = n_class
        if New:
            self.base_model = models.shufflenet_v2_x2_0(weights=ShuffleNet_V2_X2_0_Weights.DEFAULT)
            self.base_model.fc = nn.Linear(self.base_model.fc.in_features,n_class)
        else:
            self.base_model = torch.load(model_path).module.base_model
            self.base_model.fc = FC_(rate = rate, inp = self.base_model.fc.in_features, outp = n_class)
        self.base_model.fc.apply(weight_init_kaiming)
        
    def forward(self, X):
        X = self.base_model(X)
        return X
        
        
class SqueezeNet(nn.Module):
    def __init__(self, n_class, rate, New, model_path):
        super(SqueezeNet, self).__init__()
        self.n_class = n_class
        if New:
            self.base_model = models.squeezenet1_0(weights=SqueezeNet1_0_Weights.DEFAULT)
            self.base_model.classifier[1] = nn.Linear(512, n_class)
        else:
            self.base_model = torch.load(model_path).module.base_model
            self.base_model.classifier[1] = FC_(rate=rate, inp=512, outp=n_class)

    def forward(self, X):
        print("输入形状:", X.shape)
        X = self.base_model(X)
        print("经过 base_model 后的形状:", X.shape)
        return X


        
        
class SwinTransformer(nn.Module):
    def __init__(self, n_class,rate,New,model_path):
        super(SwinTransformer, self).__init__()
        self.n_class = n_class
        if New:
            self.base_model = models.swin_t(weights=Swin_T_Weights.DEFAULT)
            self.base_model.head = nn.Linear(self.base_model.head.in_features,n_class)
        else:
            self.base_model = torch.load(model_path).module.base_model
            self.base_model.head = FC_(rate = rate, inp = self.base_model.head.in_features, outp = n_class)
        self.base_model.head.apply(weight_init_kaiming)
        
    def forward(self, X):
        X = self.base_model(X)
        return X
        
       
class WideResnet50(nn.Module):
    def __init__(self, n_class,rate,New,model_path):
        super(WideResnet, self).__init__()
        self.n_class = n_class
        if New:
            self.base_model = models.wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
            self.base_model.fc = nn.Linear(self.base_model.fc.in_features,n_class)
        else:
            self.base_model = torch.load(model_path).module.base_model
            self.base_model.fc = FC_(rate = rate, inp = self.base_model.fc.in_features, outp = n_class)
        self.base_model.fc.apply(weight_init_kaiming)
        
    def forward(self, X):
        X = self.base_model(X)
        return X 
        
        
class Densenet121(nn.Module):
    def __init__(self,n_class,rate,New,model_path):
        super(Densenet121, self).__init__()
        self.n_class = n_class
        if New:
            self.base_model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
            self.base_model.classifier = nn.Linear(self.base_model.classifier.in_features,n_class)
        else:
            self.base_model = torch.load(model_path).module.base_model
            self.base_model.classifier = FC_(rate = rate, inp = self.base_model.classifier.in_features, outp = n_class)
        self.base_model.classifier.apply(weight_init_kaiming)
        
    def forward(self, X):
        X = self.base_model(X)
        return X
        
        
class Densenet161(nn.Module):
    def __init__(self,n_class,rate,New,model_path):
        super(Densenet161, self).__init__()
        self.n_class = n_class
        if New:
            self.base_model = models.densenet161(weights=DenseNet161_Weights.DEFAULT)
            self.base_model.classifier = nn.Linear(self.base_model.classifier.in_features,n_class)
        else:
            self.base_model = torch.load(model_path).module.base_model
            self.base_model.classifier = FC_(rate = rate, inp = self.base_model.classifier.in_features, outp = n_class)
        self.base_model.classifier.apply(weight_init_kaiming)
        
    def forward(self, X):
        X = self.base_model(X)
        return X 
        
        
class Densenet201(nn.Module):
    def __init__(self,n_class,rate,New,model_path):
        super(Densenet201, self).__init__()
        self.n_class = n_class
        if New:
            self.base_model = models.densenet201(weights=DenseNet201_Weights.DEFAULT)
            self.base_model.classifier = nn.Linear(self.base_model.classifier.in_features,n_class)
        else:
            self.base_model = torch.load(model_path).module.base_model
            self.base_model.classifier = FC_(rate = rate, inp = self.base_model.classifier.in_features, outp = n_class)
        self.base_model.classifier.apply(weight_init_kaiming)
        
    def forward(self, X):
        X = self.base_model(X)
        return X