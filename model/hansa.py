from typing import Dict, Union

import torch
import torch.nn.functional as F
from torch import nn

from model.hansaconv import HANSAConv


class HANSA(nn.Module):
    def __init__(self, in_channels: Union[int, Dict[str, int]],
                 out_channels: int, hidden_channels=16, heads=8, n_layers=1, metadata=None):
        super().__init__()
        self.convs = nn.ModuleList()
        self.relu = F.relu
        self.convs.append(HANSAConv(in_channels, hidden_channels, heads=heads, dropout=0.6,
                                  metadata=metadata))
        for _ in range(n_layers - 1):
            self.convs.append(HANSAConv(hidden_channels, hidden_channels, heads=heads, dropout=0.6,
                                      metadata=metadata))
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.metadata = metadata
        

    def forward(self, x_dict, edge_index_dict, labeled_class, is_return_atte=False, share_atte=None):
        if labeled_class not in x_dict or x_dict[labeled_class] is None:
            return torch.tensor([]), False
        
        now_atte = {}
        
        for i, conv in enumerate(self.convs):
            if labeled_class not in x_dict or x_dict[labeled_class] is None:
                return torch.tensor([]), False, None
            if is_return_atte:
                if share_atte is None:
                    x_dict, atte = conv(x_dict, edge_index_dict, return_semantic_attention_weights=True, share_atte=None)
                else:
                    x_dict, atte = conv(x_dict, edge_index_dict, return_semantic_attention_weights=True, share_atte=share_atte[i])
                now_atte[i] = atte
            else:
                if share_atte is None:
                    x_dict = conv(x_dict, edge_index_dict, return_semantic_attention_weights=False)
                else:
                    x_dict = conv(x_dict, edge_index_dict, return_semantic_attention_weights=False, share_atte=share_atte[i])

        x_dict = self.lin(x_dict[labeled_class])
        return x_dict, True, now_atte
 
    def get_in_channels(self):
        return self.in_channels
    
    def get_out_channels(self):
        return self.out_channels
    
    def get_metadata(self):
        return self.metadata
    
    def get_state_dict_keys(self):
        return list(self.state_dict().keys())
