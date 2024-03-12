from typing import Dict, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import HGTConv, Linear

class HGT(torch.nn.Module):
    def __init__(self, out_channels: int, hidden_channels=64, num_heads=2, num_layers=1, metadata=None, data=None):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata,
                           num_heads)
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, labeled_class):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return self.lin(x_dict[labeled_class]), True

    def get_in_channels(self):
        return self.in_channels
    
    def get_out_channels(self):
        return self.out_channels
    
    def get_metadata(self):
        return self.metadata
    
    def get_state_dict_keys(self):
        return list(self.state_dict().keys())
