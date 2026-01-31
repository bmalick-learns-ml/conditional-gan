import torch
from torch import nn
from torch.nn import functional as F

class MnistGenerator(nn.Module):
    def __init__(self, latent_dim: int = 100):
        super(MnistGenerator, self).__init__()
        self.linear_z = nn.Linear(in_features=latent_dim, out_features=200)
        self.linear_y = nn.Linear(in_features=10, out_features=1000)

        self.linear_concat = nn.Linear(in_features=1200, out_features=1200)
        self.linear_out = nn.Linear(in_features=1200, out_features=784)
        self.dropout = nn.Dropout(0.5)

    def forward(self, z, y):
        z = F.relu(self.linear_z(z))
        y = F.relu(self.linear_y(y))
        out = torch.concat((z, y), dim=1)
        out = F.relu(self.linear_concat(out))
        out = self.linear_out(self.dropout(out))
        out = F.tanh(out)
        return out
    

class Maxout(nn.Module):
    def __init__(self, in_features: int, out_features: int, pieces: int):
        super(Maxout, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features*pieces)
        self.out_features = out_features

    def forward(self, x):
        x = self.linear(x)
        x = x.view((x.size(0), self.out_features, -1))
        x,_ = torch.max(x, dim=2)
        return x
    

class MnistDiscriminator(nn.Module):
    def __init__(self):
        super(MnistDiscriminator, self).__init__()
        self.maxout_x = Maxout(in_features=784, out_features=240, pieces=5)
        self.maxout_y = Maxout(in_features=10, out_features=50, pieces=5)
        self.maxout_concat = Maxout(in_features=290, out_features=240, pieces=4)
        self.linear = nn.Linear(in_features=240, out_features=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, y):
        x = self.maxout_x(x)
        y = self.maxout_y(y)
        out = torch.concat((x, y), dim=1)
        out = self.maxout_concat(out)
        out = self.linear(self.dropout(out))
        return out