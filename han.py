import logging
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
from data_loader.hete_graph_data import all_datasets, get_data_target_node_type, load_full_dataset
from model.init import init_model

torch.cuda.empty_cache()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

for dataset in all_datasets:
    logging.info(f"Loading dataset: {dataset}")
    data = load_full_dataset(dataset, True, True)
    target_node_type = get_data_target_node_type(dataset)
    
    # # create train_loader
    # train_idx = data[target_node_type].train_mask.nonzero(as_tuple=True)[0]
    # train_loader = NeighborLoader(data, num_neighbors=[15, 15], input_nodes=(target_node_type, train_idx), input_time=None, replace=False, subgraph_type="directional", disjoint=False, temporal_strategy="uniform", time_attr=None, weight_attr=None, transform=None, transform_sampler_output=None, is_sorted=False, filter_per_worker=None,neighbor_sampler=None, batch_size=128,directed=True,shuffle=True)

    # # create test_loader
    # test_idx = data[target_node_type].test_mask.nonzero(as_tuple=True)[0]
    # test_loader = NeighborLoader(data, num_neighbors=[15, 15], input_nodes=(target_node_type, test_idx), input_time=None, replace=False, subgraph_type="directional", disjoint=False, temporal_strategy="uniform", time_attr=None, weight_attr=None, transform=None, transform_sampler_output=None, is_sorted=False, filter_per_worker=None,neighbor_sampler=None, batch_size=128,directed=True,shuffle=True)

    num_classes = data[target_node_type].y.max().item() + 1

    model = init_model('han', num_classes, data)
    data, model = data.to(device), model.to(device)

    with torch.no_grad():  # Initialize lazy modules.
        out = model(data.x_dict, data.edge_index_dict, target_node_type)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)


    def train() -> float:
        model.train()

        optimizer.zero_grad()
        data = data.to(device)
        batch_size = data[target_node_type].batch_size
        out = model(data.x_dict, data.edge_index_dict, target_node_type)

        loss = F.cross_entropy(out[target_node_type], data[target_node_type].y)
        loss.backward()
        optimizer.step()
        
        return loss.item()


    @torch.no_grad()
    def test() -> List[float]:
        model.eval()
        
        data = data.to(device)
        out = model(data.x_dict, data.edge_index_dict, target_node_type).argmax(dim=-1)
        

        # 初始化正确统计和总数统计
        correct = {split: 0 for split in ['train', 'val', 'test']}
        total = {split: 0 for split in ['train', 'val', 'test']}

        for split in ['train_mask', 'val_mask', 'test_mask']:
            mask = data['author'][split].to(device)
            total[split] += mask.sum().item()
            correct[split] += (out[mask] == data['author'].y[mask]).sum().item()

        # 计算每个分割的准确率
        accs = [correct[split] / total[split] for split in ['train', 'val', 'test']]
        
        return accs

    best_val_acc = 0
    start_patience = patience = 200
    for epoch in range(1, 200):

        loss = train()
        train_acc, val_acc, test_acc = test()
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
