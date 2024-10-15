from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# python的用法 -》 tensor数据类型
# 通过 transfrom.ToTenser去解决两个问题

# 2、为什么我们需要Tensor数据类型

# 绝对路径 G:\学术\pytorch\Project1\hymenoptera_data_practice\train\ants_image\0013035.jpg
# 相对路径 hymenoptera_data_practice/train/ants_image/0013035.jpg
img_path = "hymenoptera_data_practice/train/ants_image/0013035.jpg"
img = Image.open(img_path)
# print(img)

writer  = SummaryWriter('logs')

# 1、transform如何使用（python）
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
# print(tensor_img)

writer.add_image("Tensor Image", tensor_img)

writer.close()