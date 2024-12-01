import os
import pickle
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class MiniImageNet(Dataset):
    def __init__(self, root, train=True, transform=None, split_ratio=0.8):
        self.root = root
        self.train = train
        self.transform = transform
        self.split_ratio = split_ratio
        self.label_mapping = {}  # 用于类别前缀到整数标签的映射

        if self._check_processed():
            print('Train file has been processed' if self.train else 'Test file has been processed')
        else:
            self._process_and_split_data()

        if self.train:
            self.data, self.labels = pickle.load(
                open(os.path.join(self.root, 'processed/train.pkl'), 'rb')
            )
        else:
            self.data, self.labels = pickle.load(
                open(os.path.join(self.root, 'processed/test.pkl'), 'rb')
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

    def _check_processed(self):
        return (os.path.isfile(os.path.join(self.root, 'processed/train.pkl')) and
                os.path.isfile(os.path.join(self.root, 'processed/test.pkl')))

    def _process_and_split_data(self):
        all_images = []
        all_labels = []

        
        if not os.path.isdir(os.path.join(self.root, 'processed')):
            os.makedirs(os.path.join(self.root, 'processed'))

        # 记录类别前缀，并为每个前缀分配唯一的整数标签
        current_label = 0

        for image_file in os.listdir(self.root):
            image_path = os.path.join(self.root, image_file)
            if image_path.endswith('.jpg'):
                # 提取文件名前缀，示例：'n01532829'
                label_str = image_file.split('_')[0]
                if label_str not in self.label_mapping:
                    self.label_mapping[label_str] = current_label
                    current_label += 1
                
                label = self.label_mapping[label_str]  # 获取映射的整数标签
                img = Image.open(image_path).convert('RGB')
                img_np = np.array(img)
                
                all_images.append(img_np)
                all_labels.append(label)
        print("Unique labels:", set(all_labels))  # 检查所有标签的范围
        # 划分训练集和测试集
        data_size = len(all_images)
        indices = list(range(data_size))
        random.shuffle(indices)
        split_idx = int(data_size * self.split_ratio)
        
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]
        
        train_data = [all_images[i] for i in train_idx]
        train_labels = [all_labels[i] for i in train_idx]
        test_data = [all_images[i] for i in test_idx]
        test_labels = [all_labels[i] for i in test_idx]
        
        # 保存数据
        pickle.dump((train_data, train_labels), open(os.path.join(self.root, 'processed/train.pkl'), 'wb'))
        pickle.dump((test_data, test_labels), open(os.path.join(self.root, 'processed/test.pkl'), 'wb'))
