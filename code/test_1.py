import torch
from PIL import Image
from torchvision import transforms
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Sequential, Linear

imgs_path = "./imgs/dog.png"
image = Image.open(imgs_path)

image = image.convert('RGB')
transform = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()])

image = transform(image)
print(image.shape)

# 搭建神经网络
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = Sequential(
            Conv2d(3, 32, 5,1,2),
            MaxPool2d(2),
            Conv2d(32, 32, 5,1,2),
            MaxPool2d(2),
            Conv2d(32, 64, 5,1,2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

model = torch.load('tudui_29_gpu.pth',map_location=torch.device('cpu'))
print(model)
image = torch.reshape(image, (1,3,32,32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)
print(output.argmax(dim=1))