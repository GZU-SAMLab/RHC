import torch
from test_main import Resnet50

model = "/home/wcy/testnet/premodel/best/Resnet50_cifar100.pth"

a = torch.load(model).cuda()
print(a)
torch.save(a.module,"Resnet_for_cifar100.pth")