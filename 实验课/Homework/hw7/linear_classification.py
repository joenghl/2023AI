import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F

"""
数据集来源
https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
"""

class LinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义网络的一层(线性层), 已知有9个特征, 2个类别
        self.layer = nn.Linear(9, 2)

    def forward(self, x):
        # 经过一层线性网络后, 用softmax操作来得到各个类别的概率
        x = self.layer(x)
        return torch.softmax(x, dim=-1)


def load_data(filename):
    with open(filename, 'r') as f:
        lines = [l.rstrip() for l in f]

    num_datas = len(lines)
    num_features = 9
    datas = torch.empty([num_datas, num_features])

    # one-hot labels
    labels = torch.zeros([num_datas], dtype=torch.long)

    for i in range(num_datas):
        raw = [float(d) for d in lines[i].split(',')]
        data = torch.Tensor(raw[1:-1])
        label = torch.LongTensor([0] if raw[-1] == 2 else [1])
        datas[i] = data
        labels[i] = label
    return datas, labels


def test(data, labels, net):
    num_data = data.shape[0]
    num_correct = 0
    for i in range(num_data):
        feature = data[i]
        prob = net(feature).detach()
        dist = Categorical(prob)
        label = dist.sample().item()
        true_label = labels[i].item()
        if label == true_label:
            num_correct += 1

    return num_correct / num_data


if __name__ == "__main__":
    is_train = True
    # 读取数据, 获取数据的特征维度
    train_data, train_labels = load_data('breast-cancer-wisconsin-train.data')
    test_datas, test_labels = load_data('breast-cancer-wisconsin-test.data')
    num_data = train_data.shape[0]
    num_labels = 2
    # 定义网络. `LinearNet`类对象构造时不需要传入参数以方便批改作业
    net = LinearNet()

    if is_train:
        # 定义优化器
        sgd_optim = optim.SGD(net.parameters(), lr=1e-2)
        # 训练5000次, 每次随机取8个数据来训练(即batch_size=8)
        for i in range(5000):
            idx = torch.randint(0, num_data, [8])
            feature = train_data[idx]
            prob = net(feature)
            # 将整数型标签转化为one-hot张量, 再转为float类型
            true_label = F.one_hot(train_labels[idx], num_labels).float()
            # 计算损失值
            loss = nn.CrossEntropyLoss()(prob, true_label)
            # 梯度清零
            sgd_optim.zero_grad()
            # 反向传播, 计算梯度
            loss.backward()
            # 更新参数
            sgd_optim.step()
            # 测试当前模型
            acc = test(test_datas, test_labels, net)
            if i % 10 == 0:
                print('step: %d, loss: %.2f, accuracy: %.2f' % (i, loss.item(), acc))
        # 重要: 保存模型(训练好的神经网络参数)到当前文件夹
        torch.save(net.state_dict(), 'linear_net.pth')
    else:
        # 读取之前训练好的神经网络参数
        net.load_state_dict(torch.load('linear_net.pth'))
        # 不训练, 直接测试(批改作业时用)
        acc = test(test_datas, test_labels, net)
        print('accuracy: %.2f' % (acc))
