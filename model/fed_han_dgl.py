"""This model shows an example of federated learning environments using dgl.metapath_reachable_graph on the original heterogeneous
graph.
"""

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def get_self_weight(self, semantic_embedding):
        return self.project(semantic_embedding).mean(0)

    def forward(self, semantic_embedding, w):
        if w is None:
            w = self.get_self_weight(self, semantic_embedding)

        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand(
            (semantic_embedding.shape[0],) + beta.shape)  # (N, M, 1)

        return (beta * semantic_embedding).sum(1)  # (N, D * K)


class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : DGLGraph
        The heterogeneous graph
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(
                GATConv(
                    in_size,
                    out_size,
                    layer_num_heads,
                    dropout,
                    dropout,
                    activation=F.elu,
                    allow_zero_in_degree=True,
                )
            )
        self.semantic_attention = SemanticAttention(
            in_size=out_size * layer_num_heads
        )
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def get_semantic_embeddings(self, graph, features):
        semantic_embeddings = []

        if self._cached_graph is None or self._cached_graph is not graph:
            self._cached_graph = graph
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[
                    meta_path
                ] = dgl.metapath_reachable_graph(graph, meta_path)

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            semantic_embeddings.append(
                self.gat_layers[i](new_g, features).flatten(1))
        semantic_embeddings = torch.stack(
            semantic_embeddings, dim=1
        )  # (N, M, D * K)

        return semantic_embeddings

    def forward(self, graph, features, weight):
        semantic_embeddings = self.get_semantic_embeddings(
            self, graph, features)
        # (N, D * K)
        return self.semantic_attention(semantic_embeddings, weight)

    def get_semantic_attention_weight(self, g, h):
        semantic_embeddings = self.get_semantic_embeddings(self, g, h)
        return self.semantic_attention.get_self_weight(semantic_embeddings)


class HAN(nn.Module):
    def __init__(
        self, meta_paths, in_size, hidden_size, out_size, num_heads, dropout
    ):
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout)
        )
        for l in range(1, len(num_heads)):
            self.layers.append(
                HANLayer(
                    meta_paths,
                    hidden_size * num_heads[l - 1],
                    hidden_size,
                    num_heads[l],
                    dropout,
                )
            )
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, graph, features, weight):
        for gnn in self.layers:
            features = gnn(graph, features, weight)

        return self.predict(features)

    def get_semantic_attention_weights(self, graph, features):
        weights = []
        for gnn in self.layers:
            if isinstance(gnn, HANLayer):
                wi = gnn.get_semantic_attention_weight(graph, features)
                weights.append(wi)
        return weights
