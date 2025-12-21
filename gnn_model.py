import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
import torch.nn as nn

class DrugRepurposingHeteroGNN(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()

        self.conv1 = HeteroConv({
            ("disease", "associates", "gene"):
                SAGEConv((-1, -1), hidden_dim),

            ("gene", "rev_associates", "disease"):
                SAGEConv((-1, -1), hidden_dim),

            ("gene", "targets", "drug"):
                SAGEConv((-1, -1), hidden_dim),

            ("drug", "rev_targets", "gene"):
                SAGEConv((-1, -1), hidden_dim),
        }, aggr="sum")

        self.conv2 = HeteroConv({
            ("disease", "associates", "gene"):
                SAGEConv((hidden_dim, hidden_dim), hidden_dim),

            ("gene", "rev_associates", "disease"):
                SAGEConv((hidden_dim, hidden_dim), hidden_dim),

            ("gene", "targets", "drug"):
                SAGEConv((hidden_dim, hidden_dim), hidden_dim),

            ("drug", "rev_targets", "gene"):
                SAGEConv((hidden_dim, hidden_dim), hidden_dim),
        }, aggr="sum")

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict
