import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Sequential, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True) # 训练好的参数
vgg16_true.add_module('add_linear',nn.Linear(1000,10)) # 最后一步加一层
vgg16_true.classifier.add_module('add_linear',nn.Linear(1000,10))
print(vgg16_true)

vgg16_false.classifier[6] = nn.Linear(4096,10)
print(vgg16_false)
dataset=torchvision.datasets.CIFAR10(root='./data', train=False, download=True,transform=torchvision.transforms.ToTensor())