import torch
from torch import nn
from torch.nn import functional as F

class IRMmodel(nn.Module):
    def __init__(self, x_dim, args):
        super(IRMmodel, self).__init__()
        self.x_dim = x_dim
        self.h_dim = args.h_dim
        self.mlp = nn.Sequential(nn.Linear(x_dim, self.h_dim), nn.ReLU(),nn.Linear(self.h_dim, self.h_dim), nn.ReLU())  #
        self.pred = nn.Linear(self.h_dim, 1)

    def forward(self, x, env):
        rep = self.mlp(x)
        y_pred = self.pred(rep)
        return y_pred, rep


