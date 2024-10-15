from torch.utils.data import Dataset
from PIL import Image
import os
class Mydata(Dataset):
    # 主要提供全局变量
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, index):
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.path,img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label
    def __len__(self):
        return len(self.img_path)


if __name__ == '__main__':
    root_dir = "hymenoptera_data/hymenoptera_data/train"
    ants_dir = "ants"
    bees_dir = "bees"
    ants_dataset = Mydata(root_dir,ants_dir)
    bees_dataset = Mydata(root_dir,bees_dir)

    train_dataset = ants_dataset + bees_dataset
