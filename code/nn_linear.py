import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset=torchvision.datasets.CIFAR10(root='./data', train=False, download=True,transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64,drop_last=True)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        self.linear1 = Linear(196608,10)

    def forward(self,x):
        output = self.linear1(x)
        return output


tudui = Tudui()
for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    # output = torch.reshape(imgs,(1,1,1,-1))
    output = torch.flatten(imgs)# 展成一维的
    print(output.shape)
    output = tudui(output)
    print(output.shape)