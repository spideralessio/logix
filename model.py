from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool, global_add_pool


import torch

from torch import Tensor


class GIN(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers=3, hidden_dim=64, dropout=0.15):
        super(GIN, self).__init__()
        self.num_features, self.num_classes = num_features, num_classes
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            conv = GINConv(
                torch.nn.Sequential(
                    torch.nn.Linear(num_features if i==0 else hidden_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU()
                ),
                init_eps=1
            )
            self.convs.append(conv)
        self.fc1 = torch.nn.Linear(num_layers*3*hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, num_classes)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index, batch=None, tau=1, deterministic=False, *args, **kwargs):
        if batch is None:
            batch = torch.zeros(x.shape[0]).long().to(x.device)
        xs = []
        for conv in self.convs:
            x = conv(x, edge_index)
            xs.append(x)
            x = self.dropout(x)

        x_mean = global_mean_pool(torch.hstack(xs), batch)
        x_max = global_max_pool(torch.hstack(xs), batch)
        x_sum = global_add_pool(torch.hstack(xs), batch)
        x = torch.hstack([x_mean, x_max, x_sum])
        x = self.dropout(x)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = torch.sigmoid(self.fc2(x))
        return x

    def forward_e(self, x, edge_index, batch=None, tau=1, deterministic=False, *args, **kwargs):
        if batch is None:
            batch = torch.zeros(x.shape[0]).long().to(x.device)
        ret_x = []
        ret_y = []
        xs = []
        for conv in self.convs:
            ret_x.append(x)
            x = conv(x, edge_index)
            xs.append(x)
            ret_y.append(x)
            x = self.dropout(x)

        x_mean = global_mean_pool(torch.hstack(xs), batch)
        x_max = global_max_pool(torch.hstack(xs), batch)
        x_sum = global_add_pool(torch.hstack(xs), batch)
        x = torch.hstack([x_mean, x_max, x_sum])
        ret_x.append(x)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = torch.sigmoid(self.fc2(x))        
        ret_y.append(x)
        return ret_x, ret_y



