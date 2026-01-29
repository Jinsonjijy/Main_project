import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv


class PathogenDrugHeteroGNN(nn.Module):

    def __init__(
        self,
        num_drugs,
        num_pathways,
        num_pathogens,
        protein_dim=320,
        hidden_dim=256,
        dropout=0.3
    ):
        super().__init__()

        # Learnable embeddings
        self.drug_embedding = nn.Embedding(num_drugs, hidden_dim)
        self.pathway_embedding = nn.Embedding(num_pathways, hidden_dim)
        self.pathogen_embedding = nn.Embedding(num_pathogens, hidden_dim)

        # Protein projection
        self.protein_proj = nn.Linear(protein_dim, hidden_dim)

        # GNN Layers
        self.conv1 = HeteroConv({
            ("drug", "rev_targeted_by", "pathway"):
                SAGEConv((-1, -1), hidden_dim),

            ("pathway", "targeted_by", "drug"):
                SAGEConv((-1, -1), hidden_dim),

            ("pathway", "in_pathway", "protein"):
                SAGEConv((-1, -1), hidden_dim),

            ("protein", "rev_in_pathway", "pathway"):
                SAGEConv((-1, -1), hidden_dim),

            ("protein", "has", "pathogen"):
                SAGEConv((-1, -1), hidden_dim),

            ("pathogen", "rev_has", "protein"):
                SAGEConv((-1, -1), hidden_dim),

        }, aggr="mean")

        self.conv2 = HeteroConv({
            ("drug", "rev_targeted_by", "pathway"):
                SAGEConv((hidden_dim, hidden_dim), hidden_dim),

            ("pathway", "targeted_by", "drug"):
                SAGEConv((hidden_dim, hidden_dim), hidden_dim),

            ("pathway", "in_pathway", "protein"):
                SAGEConv((hidden_dim, hidden_dim), hidden_dim),

            ("protein", "rev_in_pathway", "pathway"):
                SAGEConv((hidden_dim, hidden_dim), hidden_dim),

            ("protein", "has", "pathogen"):
                SAGEConv((hidden_dim, hidden_dim), hidden_dim),

            ("pathogen", "rev_has", "protein"):
                SAGEConv((hidden_dim, hidden_dim), hidden_dim),

        }, aggr="mean")

        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------
    def forward(self, data):

        device = data["protein"].x.device

        # Node embeddings dictionary
        x_dict = {}

        # Protein features exist
        x_dict["protein"] = self.protein_proj(data["protein"].x)

        # Learnable embeddings for other nodes
        x_dict["drug"] = self.drug_embedding.weight
        x_dict["pathway"] = self.pathway_embedding.weight
        x_dict["pathogen"] = self.pathogen_embedding.weight

        # Layer 1
        x_dict = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {k: self.dropout(F.relu(v)) for k, v in x_dict.items()}

        # Layer 2
        x_dict = self.conv2(x_dict, data.edge_index_dict)

        return x_dict
