import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Sequential, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

vgg16 = torchvision.models.vgg16(pretrained=False)
# 保存方法1，模型结构+模型参数
torch.save(vgg16, 'vgg16_method1.pth')


# 保存方式2，模型参数（官方推荐）
torch.save(vgg16.state_dict(), 'vgg16_method2.pth')

# 陷阱
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)

    def forward(self, x):
        x = self.conv1(x)
        return x

tudui = Tudui()
torch.save(tudui, 'tudui_method1.pth')