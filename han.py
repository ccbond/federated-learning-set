import logging
from typing import Any, List

import torch
import torch.nn.functional as F
from torch import nn

import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
from data_loader.hete_graph_data import all_datasets, get_data_target_node_type, load_full_dataset
from model.init import init_model

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

    pred = model(data.x_dict, data.edge_index_dict, target_node_type).argmax(dim=-1)

    accs = []
    for split in ['train_mask', 'val_mask', 'test_mask']:
        mask = data[target_node_type][split].to(device)
        acc = ((pred[mask] == data[target_node_type].y[mask]).sum() / mask.sum())
        accs.append(acc.item())
    
    return accs


for dataset in all_datasets:
    logging.info(f"Loading dataset: {dataset}")
    data = load_full_dataset(dataset, True, True)
    target_node_type = get_data_target_node_type(dataset)
    
    num_classes = data[target_node_type].y.max().item() + 1

    model = init_model('han', num_classes, data)
    data, model = data.to(device), model.to(device)

    with torch.no_grad():  # Initialize lazy modules.
        out = model(data.x_dict, data.edge_index_dict, target_node_type)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

    best_val_acc = 0
    start_patience = patience = 200
    for epoch in range(1, 200):
        loss = train(model, data, optimizer, target_node_type)
        train_acc, val_acc, test_acc = test(model, data, target_node_type)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
            f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')

        if best_val_acc <= val_acc:
            patience = start_patience
            best_val_acc = val_acc
        else:
            patience -= 1

        if patience <= 0:
            print('Stopping training as validation accuracy did not improve '
                f'for {start_patience} epochs')
            break
