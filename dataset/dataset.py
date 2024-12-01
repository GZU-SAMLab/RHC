import torch
from torchvision import transforms
import torchvision
import torch.nn as nn
import numpy as np
from cub import cub200
import tinyimagenet as tiny
from VOC2007 import VOC2007
import os
import matplotlib.pyplot as plt
import shutil

from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
from skimage import io
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from torchvision.datasets import ImageFolder
from read_imagenet import ImageFolder_r
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class MiniImageNet(Dataset):
    def __init__(self,Path,train,transform):
        """Dataset class representing miniImageNet dataset

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if train:
            subset = 'background'
        else:
            subset = 'evaluation'
        if subset not in ('background', 'evaluation'):
            raise(ValueError, 'subset must be one of (background, evaluation)')
        self.subset = subset
        self.path = Path
        self.df = pd.DataFrame(self.index_subset(self.subset,self.path))
        print(self.df.columns)
        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

        # Setup transforms
        self.transform = transform

    def __getitem__(self, item):
        instance = Image.open(self.datasetid_to_filepath[item])
        instance = self.transform(instance)
        label = self.datasetid_to_class_id[item]
        return instance, label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(subset,path):
        """Index a subset by looping through all of its files and recording relevant information.

        # Arguments
            subset: Name of the subset

        # Returns
            A list of dicts containing information about all the image files in a particular subset of the
            miniImageNet dataset
        """
        images = []
        print('Indexing {}...'.format(subset))
        # Quick first pass to find total for tqdm bar
        subset_len = 0
        for root, folders, files in os.walk(path + '/images_{}/'.format(subset)):
            subset_len += len([f for f in files if f.endswith('.png')])

        progress_bar = tqdm(total=subset_len)
        for root, folders, files in os.walk(path + '/images_{}/'.format(subset)):
            if len(files) == 0:
                continue

            class_name = root.split('/')[-1]

            for f in files:
                progress_bar.update(1)
                images.append({
                    'subset': subset,
                    'class_name': class_name,
                    'filepath': os.path.join(root, f)
                })

        progress_bar.close()
        return images
        
def get_data(Path, img_size, batch_size, dataset_name):
  
  num_workers = 8
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
        
        
  if dataset_name == "CUB":      
      Path = Path + dataset_name
      train_data = cub200(Path, train=True, transform=transforms.Compose(train_transform_list))
      test_data = cub200(Path, train=False, transform=transforms.Compose(test_transforms_list))
      train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
      test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
      
  elif dataset_name == "cifar100":
      Path = Path + dataset_name
      cifar100_training = torchvision.datasets.CIFAR100(root=Path, train=True, download=False, transform=transforms.Compose(train_transform_list))
      cifar100_testing = torchvision.datasets.CIFAR100(root=Path, train=False, download=False, transform=transforms.Compose(test_transforms_list))
      train_loader = torch.utils.data.DataLoader(cifar100_training, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
      test_loader = torch.utils.data.DataLoader(cifar100_testing, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
      
  elif dataset_name == "VOCdevkit":
      Path = Path + dataset_name
      train_data = VOC2007(Path, train=True, transform=transforms.Compose(train_transform_list))
      test_data = VOC2007(Path, train=False, transform=transforms.Compose(test_transforms_list))
      train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
      test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
      
  elif dataset_name == "miniimagenet":
      Path = Path + "MiniImageNet_T"
      train_data = MiniImageNet(Path, train=True, transform=transforms.Compose(train_transform_list))
      test_data = MiniImageNet(Path, train=False, transform=transforms.Compose(test_transforms_list))
      train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
      test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
      
  elif dataset_name == "ImageNet-1k":
      Path = Path + "ImageNet-1k/"
      print("测试集：",end="")
      test_data = ImageFolder_r(root=Path+"val",transform=transforms.Compose(test_transforms_list))
      print("训练集：",end="")
      train_data = ImageFolder_r(root=Path+"train",transform=transforms.Compose(train_transform_list))
      train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
      test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
      
  elif dataset_name == "PDI-C":
      Path = Path + "PDI-C/"
      train_data = ImageFolder(root= Path+"train", transform=transforms.Compose(train_transform_list))  
      test_data = ImageFolder(root= Path+"test", transform=transforms.Compose(test_transforms_list))
      train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True) 
      test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)
      
  elif dataset_name == "mini":
      Path = Path + "miniimagenet/"
      train_data = ImageFolder(root= Path+"train", transform=transforms.Compose(train_transform_list))  
      test_data = ImageFolder(root= Path+"test", transform=transforms.Compose(test_transforms_list))
      train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True) 
      test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)
      
  elif dataset_name == "tiny":
      train_loader, test_loader, num_classes = tiny.load_tinyimagenet(batch_size=batch_size, nw=6)
      
  elif dataset_name == "aircraft":
      Path = Path + "fgvc-aircraft-2013b/data/dataset/"
      train_data = ImageFolder(root= Path+"trainval", transform=transforms.Compose(train_transform_list))  
      test_data = ImageFolder(root= Path+"test", transform=transforms.Compose(test_transforms_list))
      train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True) 
      test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)
      
  else:
      print("没有这个数据集:",dataset_name)
      print(1/0)
  
  
  
  return train_loader,test_loader