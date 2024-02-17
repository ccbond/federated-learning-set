import logging
from typing import Any, List
from sklearn.metrics import f1_score

import torch
import torch.nn.functional as F
from torch import nn

import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
from data_loader.hete_graph_data import all_datasets, get_data_target_node_type, load_full_dataset
from model.init import all_models, init_model

torch.cuda.empty_cache()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train(model, data, optimizer, target_node_type) -> float:
    model.train()

    optimizer.zero_grad()
    data = data.to(device)
    out = model(data.x_dict, data.edge_index_dict, target_node_type)
    mask = data[target_node_type].train_mask
    loss = F.cross_entropy(out[mask], data[target_node_type].y[mask])
    loss.backward()
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def test(model, data, target_node_type) -> List[float]:
    model.eval()

    out = model(data.x_dict, data.edge_index_dict, target_node_type)
    preds = out.argmax(dim=-1)
    labels = data[target_node_type].y

    return preds, labels

def no_fed_node_classification(model_name: str, dataset_name: str, epochs: int):
    if dataset_name == 'all':
        all_datasets = all_datasets
    else:
        all_datasets = [dataset_name]
    
    if model_name == 'all':
        model_types = all_models
    else:
        model_types = [model_name]

    for model_type in model_types:
        for dataset in all_datasets:
            logging.info(f"Loading dataset: {dataset}")
            data = load_full_dataset(dataset, True, True)
            target_node_type = get_data_target_node_type(dataset)
            
            num_classes = data[target_node_type].y.max().item() + 1

            model = init_model(model_type, num_classes, data)
            data, model = data.to(device), model.to(device)

            with torch.no_grad():  # Initialize lazy modules.
                out = model(data.x_dict, data.edge_index_dict, target_node_type)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

            for epoch in range(1, epochs):
                loss = train(model, data, optimizer, target_node_type)
                preds, labels = test(model, data, target_node_type)

                preds = preds.cpu().numpy()
                labels = labels.cpu().numpy()

                macro_f1 = f1_score(labels, preds, average='macro')
                micro_f1 = f1_score(labels, preds, average='micro')

                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}')

            logging.info(f'DataSet: {dataset}, Model: {model_type}, Loss: {loss}, Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}, Epochs: {epoch}')
