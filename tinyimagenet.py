from typing import Any
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import glob
import argparse
from PIL import Image


parser = argparse.ArgumentParser("parameters")
parser.add_argument("--batch-size", type=int, default=120, help="number of batch size, (default, 512)")
parser.add_argument('--workers', type=int, default=7)

parser.add_argument('--seed', default=42, type=int, nargs='+',
                help='seed for initializing training. ')

args = parser.parse_args()
 
class TrainTinyImageNet(Dataset):
    def __init__(self, root, id, transform=None) -> None:
        super().__init__()
        self.filenames = glob.glob(root + "/train/*/*/*.JPEG")
        self.transform = transform
        self.id_dict = id
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx: Any) -> Any:
        img_path = self.filenames[idx]
        image = Image.open(img_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        label = self.id_dict[img_path.split('/')[-3]]
        if self.transform:
            image = self.transform(image)
        return image, label
 
class ValTinyImageNet(Dataset):
    def __init__(self, root, id, transform=None):
        self.filenames = glob.glob(root + "/val/images/*.JPEG")
        self.transform = transform
        self.id_dict = id
        self.cls_dic = {}
        for i, line in enumerate(open(root + '/val/val_annotations.txt', 'r')):
            a = line.split('\t')
            img, cls_id = a[0], a[1]
            self.cls_dic[img] = self.id_dict[cls_id]
 
    def __len__(self):
        return len(self.filenames)
 
    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = Image.open(img_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        label = self.cls_dic[img_path.split('/')[-1]]
        if self.transform:
            image = self.transform(image)
        return image, label
 
def load_tinyimagenet(batch_size,nw):
    batch_size = args.batch_size
    nw = args.workers
    root = '/home/zaowuyu/dataset/tinyimagenet/tiny-imagenet-200'
    id_dic = {}
    for i, line in enumerate(open(root+'/wnids.txt','r')):
        id_dic[line.replace('\n', '')] = i
    num_classes = len(id_dic)
    data_transform = {
        "train": transforms.Compose([transforms.Resize(224),
                                     transforms.RandomCrop(224, padding=4),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    train_dataset = TrainTinyImageNet(root, id=id_dic, transform=data_transform["train"])
    val_dataset = ValTinyImageNet(root, id=id_dic, transform=data_transform["val"])
 
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw)
    
    print("TinyImageNet Loading SUCCESS"+
          "\nlen of train dataset: "+str(len(train_dataset))+
          "\nlen of val dataset: "+str(len(val_dataset)))
    
    return train_loader, val_loader, num_classes