import os.path as osp

import torch
import torch.nn.functional as F
from torch import nn

import torch_geometric.transforms as T
from torch_geometric.datasets import DBLP, AMiner, OGB_MAG, MovieLens, IMDB, Taobao, AmazonBook, LastFM, HGBDataset
from torch_geometric.nn import HANConv

from data_loader import load_acm

# all_datasets = ["DBLP", "IMDB", "OGB_MAG", "LastFM" "AmazonBook", "MovieLens", "AMiner",  "Taobao", "HGB-ACM", "HGB-DBLP", "HGB-Freebase", "HGB-IMDB"]
all_datasets = ["ACM", "DBLP", "IMDB"]


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

        metapaths = [[("author", "paper"),("paper", "term"),("term", "paper"),("paper", "author")],
                [("author", "paper"), ("paper", "author")],[("author", "paper"),("paper", "conference"),("conference", "paper"),("paper", "author")]]
        # metapaths = [[("paper", "term"), ("term", "paper")],
        #     [("author", "paper"), ("paper", "author")]]
            
        transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=drop_orig_edge_types,
                                drop_unconnected_node_types=drop_unconnected_node_types)
        dataset = DBLP(path, transform=transform, )
        data = dataset[0]
    
    elif data_name == "HGB-ACM":
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../data/HGBDataset/ACM')
        metapaths = [[('paper', 'author'), ('author', 'paper')],
                    [('paper', 'subject'), ('subject', 'paper')]]
        transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=drop_orig_edge_types,
                                drop_unconnected_node_types=drop_unconnected_node_types)
        dataset = HGBDataset(path, "acm")
        data = dataset[0]

    elif data_name == "HGB-DBLP":
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../data/HGBDataset/DBLP')
        metapaths = [[('author', 'paper'), ('paper', 'author')],
                    [('author', 'paper'),('paper', 'term'),('term', 'paper'), ('paper', 'author')],
                    [('author', 'paper'),('paper', 'venue'),('venue', 'paper'), ('paper', 'author')]]
        transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=drop_orig_edge_types,
                                drop_unconnected_node_types=drop_unconnected_node_types)
        dataset = HGBDataset(path, "dblp", transform=transform)
        data = dataset[0]

    elif data_name == "HGB-Freebase":
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../data/HGBDataset/Freebase')
        # metapaths = [[('movie', 'actor'), ('actor', 'movie')],
        #             [('movie', 'director'), ('director', 'movie')]]
        # transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=drop_orig_edge_types,
        #                         drop_unconnected_node_types=drop_unconnected_node_types)
        dataset = HGBDataset(path, "freebase")
        data = dataset[0]
    elif data_name == "HGB-IMDB":
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../data/HGBDataset/IMDB')
        metapaths = [[('movie', 'actor'), ('actor', 'movie')],
                    [('movie', 'director'), ('director', 'movie')]]
        transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=drop_orig_edge_types,
                                drop_unconnected_node_types=drop_unconnected_node_types)
        dataset = HGBDataset(path, "imdb", transform=transform)
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
        # metapaths = [[('author', 'paper'), ('paper', 'paper'),('paper', 'author')],
        #             [('author', 'paper'), ('paper', 'field_of_study'), ('field_of_study', 'paper'), ('paper', 'author')],
        #             [('author', 'paper'), ('paper', 'author')],
        #             [('author', 'institution'), ('institution', 'author')]]
        metapaths = [[('author', 'paper'), ('paper', 'paper')],
                    [('author', 'paper'), ('paper', 'field_of_study')]]

        transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=drop_orig_edge_types,
                                drop_unconnected_node_types=drop_unconnected_node_types)
        dataset = OGB_MAG(path, transform=transform, preprocess='metapath2vec')
        data = dataset[0]
        
    elif data_name == "LastFM":
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../data/LastFM')
        metapaths = [[('user', 'artist'), ('artist', 'user')], [('user', 'artist'),('artist', 'tag'), ('tag', 'artist'), ('artist', 'user')],
                     [('artist', 'tag'), ('tag', 'artist')], [('artist', 'user'), ('user', 'artist')], [('artist', 'user'),('user', 'user'), ('user', 'artist')]]
        transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=drop_orig_edge_types,
                                   drop_unconnected_node_types=drop_unconnected_node_types)
        dataset = LastFM(path, transform=transform)
        data = dataset[0] 
    
    elif data_name == "MovieLens":
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../data/MovieLens')
        metapaths = []
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
        metapaths = [[('paper', 'author'), ('author', 'paper')],[('paper', 'venue'), ('venue', 'paper')]]
        transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=drop_orig_edge_types,
                                drop_unconnected_node_types=drop_unconnected_node_types)
        dataset = AMiner(path, transform=transform)
        data = dataset[0]
        
    elif data_name == "ACM":
        data = load_acm.load_acm_data()

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
    elif dataset == "AMiner":
        return "author"
    elif dataset == "ACM":
        return "paper"
    elif dataset == "HGB-DBLP":
        return "author"
    elif dataset == "HGB-Freebase":
        return "book"
    elif dataset == "HGB-IMDB":
        return "movie"
    else:
        return None

def get_is_need_mini_batch(dataset: str):
    if dataset == "DBLP":
        return True
    elif dataset == "IMDB":
        return True
    elif dataset == "OGB_MAG":
        return True
    elif dataset == "AMiner":
        return True
    elif dataset == "ACM":
        return True
    elif dataset == "HGB-IMDB":
        return True
    elif dataset == "HGB-DBLP":
        return True 
    else:
        return False

def get_batch_size_list(dataset: str):
    if dataset == "ACM":
        return [300, 184, 92]
    elif dataset == "DBLP":
        return [40, 90, 150]
    elif dataset == "IMDB":
        return [140, 40, 90]
    else:
        return [128]