import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv


class BacteriaSpecificHeteroGNN(nn.Module):
    def __init__(self, input_dim=640, hidden_dim=256, dropout=0.3):
        super().__init__()

        self.proj = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.conv1 = HeteroConv({
            ("disease", "associated_with", "protein"):
                SAGEConv((-1, -1), hidden_dim),
            ("protein", "rev_associated_with", "disease"):
                SAGEConv((-1, -1), hidden_dim),

            ("protein", "targeted_by", "drug"):
                SAGEConv((-1, -1), hidden_dim),
            ("drug", "rev_targeted_by", "protein"):
                SAGEConv((-1, -1), hidden_dim),
        }, aggr="mean")

        self.conv2 = HeteroConv({
            ("disease", "associated_with", "protein"):
                SAGEConv((hidden_dim, hidden_dim), hidden_dim),
            ("protein", "rev_associated_with", "disease"):
                SAGEConv((hidden_dim, hidden_dim), hidden_dim),

            ("protein", "targeted_by", "drug"):
                SAGEConv((hidden_dim, hidden_dim), hidden_dim),
            ("drug", "rev_targeted_by", "protein"):
                SAGEConv((hidden_dim, hidden_dim), hidden_dim),
        }, aggr="mean")

    def forward(self, x_dict, edge_index_dict):

        x_dict = {k: F.relu(self.proj(v)) for k, v in x_dict.items()}

        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: self.dropout(F.relu(v)) for k, v in x_dict.items()}

        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: F.normalize(v, dim=1) for k, v in x_dict.items()}

        return x_dict