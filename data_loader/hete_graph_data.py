import os.path as osp

import torch
import torch.nn.functional as F
from torch import nn

import torch_geometric.transforms as T
from torch_geometric.datasets import DBLP, AMiner, OGB_MAG, MovieLens, IMDB, Taobao, AmazonBook
from torch_geometric.nn import HANConv

all_datasets = ["DBLP", "IMDB", "OGB_MAG", "AmazonBook", "MovieLens", "AMiner",  "Taobao"]


def add_zero_features_for_graph(data, feature_dim = 128):
    for node_type in data.node_types:
        if not hasattr(data[node_type], 'x') or data[node_type].x is None:
            num_nodes = data[node_type].num_nodes
            data[node_type].x = torch.zeros((num_nodes, feature_dim), dtype=torch.float)
    return data


def load_full_dataset(data_name: str, drop_orig_edge_types: bool, drop_unconnected_node_types: bool, add_zero_features: bool = False): 
    data = None
    
    if data_name == "DBLP":
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../data/DBLP')
        metapaths = [[("paper", "term"), ("term", "paper")],
                     [("author", "paper"), ("paper", "author")]]
        transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=drop_orig_edge_types,
                                drop_unconnected_node_types=drop_unconnected_node_types)
        dataset = DBLP(path, transform=transform, )
        data = dataset[0]
    
    elif data_name == "IMDB":
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../data/IMDB')
        metapaths = [[('movie', 'actor'), ('actor', 'movie')],
                    [('movie', 'director'), ('director', 'movie')]]
        transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=drop_orig_edge_types,
                                drop_unconnected_node_types=drop_unconnected_node_types)
        dataset = IMDB(path, transform=transform)
        data = dataset[0]

    elif data_name == "OGB_MAG":
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../data/OGB_MAG')
        metapaths = [[('author', 'paper'), ('paper', 'paper')],
                    [('paper', 'paper'), ('paper', 'field_of_study')]]
        transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=drop_orig_edge_types,
                                drop_unconnected_node_types=drop_unconnected_node_types)
        dataset = OGB_MAG(path, transform=transform, preprocess='metapath2vec')
        data = dataset[0]
    
    elif data_name == "MovieLens":
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../data/MovieLens')
        metapaths = [[('author', 'paper'), ('paper', 'paper')],
                    [('author', 'paper'), ('paper', 'field_of_study')]]
        transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=drop_orig_edge_types,
                                drop_unconnected_node_types=drop_unconnected_node_types)
        dataset = MovieLens(path, transform=transform)
        data = dataset[0]
    
    elif data_name == "Taobao":
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../data/Taobao')
        metapaths = [[('item', 'user'), ('user', 'item')],
                    [('item', 'category'), ('category', 'item')]]
        transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=drop_orig_edge_types,
                                drop_unconnected_node_types=drop_unconnected_node_types)
        dataset = Taobao(path, transform=transform)
        data = dataset[0]
    
    elif data_name == "AmazonBook":
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../data/AmazonBook')
        metapaths = [[('book', 'user'), ('user', 'book')]]
        transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=drop_orig_edge_types,
                                drop_unconnected_node_types=drop_unconnected_node_types)
        dataset = AmazonBook(path, transform=transform)
        data = dataset[0]
    
    elif data_name == "AMiner":
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../data/AMiner')
        metapaths = [[('author', 'paper'), ('paper', 'author')],
                    [('paper', 'venue'), ('venue', 'paper')]]
        transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=drop_orig_edge_types,
                                drop_unconnected_node_types=drop_unconnected_node_types)
        dataset = AMiner(path, transform=transform)
        data = dataset[0]

    if add_zero_features:
        data = add_zero_features_for_graph(data)

    return data

def get_data_target_node_type(dataset: str):
    if dataset == "DBLP":
        return "author"
    elif dataset == "IMDB":
        return "movie"
    elif dataset == "OGB_MAG":
        return "paper"
    else:
        return None
