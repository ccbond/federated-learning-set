import os
import random
import torch
import logging
from utils.args import args


def load_hete_graph_data(dataset_name, device, data_dir='/data/DBLP'):
    """
    Load the heterogeneous graph data.

    Parameters:
    - dataset_name (str): Name of the dataset.
    - device (torch.device): Device to store the data.
    - data_dir (str): Path to the directory where the data is stored.

    Returns:
    - HeteroData: The heterogeneous graph data.
    - int: Number of classes.
    - Tensor: Training mask.
    - Tensor: Validation mask.
    - Tensor: Test mask.
    - Tensor: Labels.
    """

    path = os.path.abspath(os.getcwd() + data_dir)

    if dataset_name == 'DBLP':
        from torch_geometric.datasets import DBLP
        logging.info("Data path = %s" % path)

        dataset = DBLP(path)
        graph = dataset[0]
        logging.info("Load succeed! DBLP graph:\n %s" % graph)

        num_classes = torch.max(graph['author'].y).item() + 1
        graph['conference'].x = torch.ones((graph['conference'].num_nodes, 1))
        graph = graph.to(device)
        train_mask, val_mask, test_mask = graph['author'].train_mask, graph[
            'author'].val_mask, graph['author'].test_mask
        y = graph['author'].y
        logging.info("Loaded DBLP dataset succeed. Number of nodes: %d, number of edges: %d, number of classes: %d." % (
            graph.num_nodes, graph.num_edges, num_classes))

        return graph, num_classes, train_mask, val_mask, test_mask, y

    else:
        logging.info("Dataset not found.")
        return None


def load_fed_hete_graph_data(dataset_name, device, data_dir='/data/DBLP'):
    path = os.path.abspath(os.getcwd() + data_dir)

    if dataset_name == 'DBLP':
        from torch_geometric.datasets import DBLP

        logging.info("data path = %s" % path)
        dataset = DBLP(path)
        graph = dataset[0]

        # sub graph index
        author_nodes_indexs = list(range(4056))
        paper_nodes_indexs = list(range(14327))

        random.shuffle(author_nodes_indexs)
        random.shuffle(paper_nodes_indexs)

        author_quarter_length = len(author_nodes_indexs) // args.client_nums
        paper_quarter_length = len(paper_nodes_indexs) // args.client_nums

        author_subgraph_indexs = []
        paper_subgraph_indexs = []
        for i in range(args.client_nums):
            if i != args.client_nums - 1:
                author_subgraph_indexs.append(
                    author_nodes_indexs[i * author_quarter_length: (i + 1) * author_quarter_length])
                paper_subgraph_indexs.append(
                    paper_nodes_indexs[i * paper_quarter_length: (i + 1) * paper_quarter_length])
            else:
                author_subgraph_indexs.append(
                    author_nodes_indexs[i * author_quarter_length: len(author_nodes_indexs)])
                paper_subgraph_indexs.append(
                    paper_nodes_indexs[i * paper_quarter_length: len(paper_nodes_indexs)])

        subgraphs = []
        num_classes = []
        train_masks = []
        test_masks = []
        val_masks = []
        y = []
        features = []

        for i in range(args.client_nums):
            sub_graph = graph.subgraph({
                'author': torch.tensor(author_subgraph_indexs[i], dtype=torch.long),
                'paper': torch.tensor(paper_subgraph_indexs[i], dtype=torch.long)
            })

            num_class = torch.max(sub_graph['author'].y).item() + 1
            sub_graph['conference'].x = torch.ones(
                (sub_graph['conference'].num_nodes, 1))
            sub_graph = sub_graph.to(device)
            subgraphs.append(sub_graph)
            train_mask, val_mask, test_mask = sub_graph['author'].train_mask, sub_graph[
                'author'].val_mask, sub_graph['author'].test_mask
            sub_y = sub_graph['author'].y
            features = sub_graph['author'].x
            num_classes.append(num_class)
            train_masks.append(train_mask)
            val_masks.append(val_mask)
            test_masks.append(test_mask)
            y.append(sub_y)
            features.append(features)

        logging.info("Loading Sub graphs succeed! \n %s" % subgraphs)

        return subgraphs, num_classes, train_masks, test_masks, val_masks, y, features

    else:
        logging.info('Dataset not found.')
        return None
