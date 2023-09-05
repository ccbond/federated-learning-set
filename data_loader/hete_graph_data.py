import os
import torch
import logging


def load_hete_graph_data(dataset_name, device, data_dir='/data'):
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

        logging.info("data path = %s" % path)

        dataset = DBLP(path)
        graph = dataset[0]
        num_classes = torch.max(graph['author'].y).item() + 1
        graph['conference'].x = torch.ones((graph['conference'].num_nodes, 1))
        graph = graph.to(device)
        train_mask, val_mask, test_mask = graph['author'].train_mask, graph[
            'author'].val_mask, graph['author'].test_mask
        y = graph['author'].y

        logging.info("Loaded DBLP dataset. Number of nodes: %d, number of edges: %d, number of classes: %d" % (
            graph.num_nodes, graph.num_edges, num_classes))

        return graph, num_classes, train_mask, val_mask, test_mask, y

    else:
        logging.info("Dataset not found.")
        return None
