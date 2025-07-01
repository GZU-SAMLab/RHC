import torch
from torchvision import transforms
import torch.nn as nn
from cub import cub200
from torch.utils.data import DataLoader

# 用于加载PIL图片的模块
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_data(Path, img_size, batch_size, dataset_name):
    num_workers = 8
    
    # 训练和测试的图像变换
    train_transform_list = [
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ]
    
    test_transforms_list = [
        transforms.Resize(int(img_size / 0.875)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ]
    
    # 如果数据集是 CUB (Caltech-UCSD Birds 200)
    if dataset_name == "CUB":
        Path = Path + dataset_name
        train_data = cub200(Path, train=True, transform=transforms.Compose(train_transform_list))
        test_data = cub200(Path, train=False, transform=transforms.Compose(test_transforms_list))
        
        # 创建 DataLoader
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    else:
        print("没有这个数据集:", dataset_name)
        print(1 / 0)  # 结束程序

    return train_loader, test_loader
