import torch.nn as nn
import torch.optim
from torch.autograd import Variable
from .cnn_network import CnnNetwork
from .dataset import data_loader


def train_model():
    model = CnnNetwork()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    error = nn.MultiLabelSoftMarginLoss()

    for i in range(5):
        for (batch_x, batch_y) in data_loader():
            optimizer.zero_grad()
            image = Variable(batch_x)
            label = Variable(batch_y)
            out = model(image)
            loss = error(out, label)
            print(loss)
            loss.backward()
            optimizer.step()
    # 保存模型的名字 model1.pth
    torch.save(model.state_dict(), "model.pkl")


if __name__ == '__main__':
    train_model()