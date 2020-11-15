# -*- coding: UTF-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
from .dataset import transform
from PIL import Image
from .cnn_network import CnnNetwork
from .image_setting import out_place, out_length


def main(image: str) -> None:
    cnn = CnnNetwork()
    cnn.eval()
    # 加载这个模型model1.pkl
    cnn.load_state_dict(torch.load('model.pkl'))
    print("load cnn net.")
    t_image = transform(Image.open(image).convert("RGB"))
    v_image = Variable(t_image)
    predict_label = cnn(v_image)
    c0 = out_place[np.argmax(predict_label[0, 0:out_length].data.numpy())]
    c1 = out_place[np.argmax(predict_label[0, out_length:2 * out_length].data.numpy())]
    c2 = out_place[np.argmax(predict_label[0, 2 * out_length:3 * out_length].data.numpy())]
    c3 = out_place[np.argmax(predict_label[0, 3 * out_length:4 * out_length].data.numpy())]
    predict_label = '%s%s%s%s' % (c0, c1, c2, c3)
    print(f"当前验证码为:{predict_label}")


if __name__ == '__main__':
    main("0kxr.jpeg")
