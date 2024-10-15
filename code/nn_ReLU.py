import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d, ReLU
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                      [-1, 3]])

input = torch.reshape(input,(-1,1,2,2))
print(input.shape)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        self.relu1 = ReLU()

    def forward(self,input):
        output = self.relu1(input)
        return output

todui = Tudui()
output = todui(input)
print(output)