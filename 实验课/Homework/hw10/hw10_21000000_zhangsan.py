import torch
import torch.nn as nn

device = 'cpu'


class Net1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(4, 2)

    def forward(self, x):
        return self.l1(x)


class Net2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(4, 1)

    def forward(self, x):
        return self.l1(x) + 1


class MyAgent:
    def __init__(self) -> None:
        self.net1 = Net1().to(device)
        self.net2 = Net2().to(device)

    def get_action(self, state, eval_mode=False):
        v1 = self.net1(torch.Tensor(state)).mean()
        v2 = self.net2(torch.Tensor(state))
        return 1 if v1 + v2 > 0 else 0

    def load_model(self, file_name):
        self.net1.load_state_dict(torch.load(file_name + "_net1.pth"))
        self.net2.load_state_dict(torch.load(file_name + "_net2.pth"))


if __name__ == '__main__':
    print("hello")
