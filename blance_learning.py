import torch
import torch.nn as nn
from wcy_method import doubleSGD

class BL(nn.Module):
    def __init__(self,image_size,label_size,count):
        super(BL,self).__init__()
        self.img = torch.randn([1,3,image_size,image_size]).cuda()
        self.label = torch.ones([1,label_size]).cuda()
        self.label = self.label/label_size
        self.count = count
        print("一共进行",count,"次")
        print("使用平衡训练法")
        
    def forward(self,net):
        solver = doubleSGD(net.parameters(), lr=0.001, momentum=0.9,weight_decay=1e-4)
        for i in range(self.count):
           solver.zero_grad()
           output = net(self.img)
           loss = self.label-output
           loss = loss**2
           loss = loss.sum()
           loss = loss**0.5
           loss.backward()
           solver.step(0.5)
        return net
        
        