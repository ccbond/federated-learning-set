import logging

from sklearn.metrics import f1_score

import torch
import torch.nn.functional as F
from torch import nn

import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
from data_loader.hete_graph_data import all_datasets, get_data_target_node_type, get_is_need_mini_batch, load_full_dataset
from model.init import all_models, init_model
from train.train import no_fed_test_nc, no_fed_train_nc

torch.cuda.empty_cache()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def no_fed_node_classification(model_name: str, dataset_name: str, epochs: int):
    if dataset_name == 'all':
        datasets = all_datasets
    else:
        datasets = [dataset_name]
    
    if model_name == 'all':
        models = all_models
    else:
        models = [model_name]
        
    is_mini_batch = get_is_need_mini_batch(dataset_name)
    print(f"Is mini batch: {is_mini_batch}")

    for model_type in models:
        for dataset in datasets:
            logging.info(f"Loading dataset: {dataset}")
            data = load_full_dataset(dataset, True, True, True)
            logging.info(data)
            target_node_type = get_data_target_node_type(dataset)
            
            num_classes = data[target_node_type].y.max().item() + 1

            model = init_model(model_type, num_classes, data)
            data, model = data.to(device), model.to(device)

            with torch.no_grad():  # Initialize lazy modules.                    
                model(data.x_dict, data.edge_index_dict, target_node_type)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

            for epoch in range(1, epochs):
                loss = no_fed_train_nc(model, data, optimizer, target_node_type, is_mini_batch, device)
                preds, labels = no_fed_test_nc(model, data, target_node_type, is_mini_batch, device)

                preds = preds.cpu().numpy()
                labels = labels.cpu().numpy()

                macro_f1 = f1_score(labels, preds, average='macro')
                micro_f1 = f1_score(labels, preds, average='micro')

                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}')

            logging.info(f'DataSet: {dataset}, Model: {model_type}, Loss: {loss}, Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}, Epochs: {epoch}')
