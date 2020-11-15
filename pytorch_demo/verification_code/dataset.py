import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import glob
from .image_setting import img_path, out_length, word_length, out_place
from torchvision import transforms as T
from PIL import Image


def read_data():
    """
    读取图片样本
    :return: 样本路径, 样本名
    """
    train_data = glob.glob(f'{img_path}\\*.jpeg')
    train_label = np.array(
        [train_data[index].split('\\')[-1].split('.')[0].split('_')[0] for index in
         range(len(train_data))])
    return train_data, train_label


def one_hot(x):
    """
    :param x: 图面样本名
    :return: 样本的 arrary数据
    """
    tmp = [0 for i in range(out_length*word_length)]
    for step, i in enumerate(x):
        tmp[out_place.index(i.lower()) + out_length * step] = 1
    return np.array(tmp)


transform = T.Compose([
    # T.Resize((160, 80)),
    T.ToTensor(),
    # T.Normalize(std=[0.5, 0.5, 0.5], mean=[0.5, 0.5, 0.5])
])


class DataSet(Dataset):
    def __init__(self,):
        self.image_path, self.label = read_data()

    def __getitem__(self, index):
        path = self.image_path[index]
        img = transform(Image.open(path).convert("RGB"))
        label = torch.from_numpy(one_hot(self.label[index])).float()
        return img, label

    def __len__(self):
        return len(self.image_path)


def data_loader():
    data_set = DataSet()
    data_load = DataLoader(data_set, batch_size=64, shuffle=True, drop_last=True)
    return data_load
