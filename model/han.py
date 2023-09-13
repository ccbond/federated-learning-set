import logging
from torch import nn
from torch_geometric.nn import HANConv


class HAN(nn.Module):
    """
    Heterogeneous Graph Attention Network (HAN) model.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, graph):
        """
        Initialize the HAN model.

        Parameters:
        - in_channels (int): Number of input features.
        - hidden_channels (int): Number of hidden units.
        - out_channels (int): Number of output features.
        - graph (HeteroData): Input graph.
        """
        super(HAN, self).__init__()
        self.conv1 = HANConv(in_channels, hidden_channels,
                             graph.metadata(), heads=1)
        self.conv2 = HANConv(hidden_channels, out_channels,
                             graph.metadata(), heads=1)

    def forward(self, data):
        """
        Forward pass for the HAN model.

        Parameters:
        - data (HeteroData): Input data.

        Returns:
        - Tensor: Output data.
        """
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        logging.info(f"x_dict: {x_dict}")
        logging.info(f"edge_index_dict: {edge_index_dict}")
        x = self.conv1(x_dict, edge_index_dict)
        x = self.conv2(x, edge_index_dict)
        x = x['author']

        return x
