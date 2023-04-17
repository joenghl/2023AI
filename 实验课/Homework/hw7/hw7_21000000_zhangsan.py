import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class MyConvNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


if __name__=="__main__":
    train_data = torch.load('train_data.pth')
    train_labels = torch.load('train_labels.pth')
    test_data = torch.load('test_data.pth')
    test_labels = torch.load('test_labels.pth')

    # 定义网络. `MyConvNet`类对象构造时不需要传入参数以方便批改作业
    net = MyConvNet()

    # 如果已经训练完成, 则直接读取网络参数. 注意文件名改为自己的信息
    net.load_state_dict(torch.load('hw7_21000000_zhangsan.pth'))
