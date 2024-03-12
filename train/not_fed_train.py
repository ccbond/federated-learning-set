import logging
import time
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
import torch
from data_loader.hete_graph_data import all_datasets, get_batch_size_list, get_data_target_node_type, get_is_need_mini_batch, load_full_dataset
from model.show import show_model
from model.init import all_models, init_model
from train.train import no_fed_test_nc, no_fed_train_nc, select_last_batch_fed_train_nc

torch.cuda.empty_cache()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def no_fed_node_classification(model_name: str, dataset_name: str):
    if dataset_name == 'all':
        datasets = all_datasets
    else:
        datasets = [dataset_name]
    
    if model_name == 'all':
        models = all_models
    else:
        models = [model_name]

    for model_type in models:
        for dataset in datasets:
            # batch_size_list = get_batch_size_list(dataset)
            # is_mini_batch = get_is_need_mini_batch(dataset)
            batch_size_list = [128]
            is_mini_batch = False

            data = load_full_dataset(dataset, True, True, True)
            # logging.info(data)
            target_node_type = get_data_target_node_type(dataset)
            
            for batch_size in batch_size_list:
                ma_f1_list = []
                mi_f1_list = []
                print(f'Start train model: {model_type}, dataset: {dataset}, batch_size: {batch_size}')
                logging.info(f"Loading dataset: {dataset}, mini batch: {is_mini_batch}, model: {model_type}, batch_size: {batch_size}")
                for i in tqdm(range(10)):
                    num_classes = int(data[target_node_type].y.max().item() + 1)

                    model = init_model(model_type, num_classes, data)
                    data, model = data.to(device), model.to(device)

                    with torch.no_grad():  # Initialize lazy modules.                    
                        model(data.x_dict, data.edge_index_dict, target_node_type)

                    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

                    epoch = 0
                    best_macri_fl = 0
                    start_patience = patience = 50

                    epoch_start_time = time.time()
                    
                    epoch_end_time_data = []

                    while True:
                        epoch += 1
                        
                        # print(f'Epoch: {epoch}')
                        loss = no_fed_train_nc(model, data, optimizer, target_node_type, is_mini_batch, batch_size, device)
                        preds, labels = no_fed_test_nc(model, data, target_node_type, is_mini_batch, batch_size, device)

                        preds = preds.cpu().numpy()
                        labels = labels.cpu().numpy()

                        macro_f1 = f1_score(labels, preds, average='macro')
                        micro_f1 = f1_score(labels, preds, average='micro')

                        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}')

                        if best_macri_fl <= macro_f1:
                            patience = start_patience
                            best_macri_fl = macro_f1
                        else:
                            patience -= 1
                        
                        epoch_end_time = time.time()
                        time_interval = epoch_end_time - epoch_start_time
                        epoch_end_time_data.append(time_interval)
                            

                        if patience <= 0:
                            print('Stopping training as validation accuracy did not improve '
                                f'for {start_patience} epochs')
                            break

                    logging.info(f'Index: {i}, DataSet: {dataset}, Model: {model_type}, Loss: {loss}, Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}, Epochs: {epoch}')
                    if i == 9:
                        logging.info(f'Epoch time:\n {epoch_end_time_data}')
                    
                    ma_f1_list.append(macro_f1)
                    mi_f1_list.append(micro_f1)
                    
                    show_model(model)
                    
                avg_ma_f1 = np.mean(ma_f1_list)
                avg_mi_f1 = np.mean(mi_f1_list)
                logging.info(f'DataSet: {dataset}, Model: {model_type}, Avg Macro F1: {avg_ma_f1:.4f}, Avg Micro F1: {avg_mi_f1:.4f}')


def no_fed_node_classification_select_one(model_name: str, dataset_name: str):
    if dataset_name == 'all':
        datasets = all_datasets
    else:
        datasets = [dataset_name]
    
    if model_name == 'all':
        models = all_models
    else:
        models = [model_name]

    for model_type in models:
        for dataset in datasets:
            batch_size_list = get_batch_size_list(dataset)
            is_mini_batch = get_is_need_mini_batch(dataset)
            # batch_size_list = [128]
            # is_mini_batch = False

            data = load_full_dataset(dataset, True, True, True)
            # logging.info(data)
            target_node_type = get_data_target_node_type(dataset)
            
            for batch_size in batch_size_list:
                ma_f1_list = []
                mi_f1_list = []
                print(f'Start train model: {model_type}, dataset: {dataset}, batch_size: {batch_size}')
                logging.info(f"Loading dataset: {dataset}, mini batch: {is_mini_batch}, model: {model_type}, batch_size: {batch_size}")
                for i in tqdm(range(1)):
                    num_classes = int(data[target_node_type].y.max().item() + 1)

                    model = init_model(model_type, num_classes, data)
                    data, model = data.to(device), model.to(device)

                    with torch.no_grad():  # Initialize lazy modules.                    
                        model(data.x_dict, data.edge_index_dict, target_node_type)

                    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

                    epoch = 0
                    best_macri_fl = 0
                    start_patience = patience = 50
                    
                    
                    epoch_start_time = time.time()
                    
                    epoch_end_time_data = []

                    while True:
                        epoch += 1
                        loss = select_last_batch_fed_train_nc(model, data, optimizer, target_node_type, is_mini_batch, batch_size, device)
                        preds, labels = no_fed_test_nc(model, data, target_node_type, is_mini_batch, batch_size, device)

                        preds = preds.cpu().numpy()
                        labels = labels.cpu().numpy()

                        macro_f1 = f1_score(labels, preds, average='macro')
                        micro_f1 = f1_score(labels, preds, average='micro')

                        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}')

                        if best_macri_fl <= macro_f1:
                            patience = start_patience
                            best_macri_fl = macro_f1
                        else:
                            patience -= 1
                            
                        epoch_end_time = time.time()
                        time_interval = epoch_end_time - epoch_start_time
                        epoch_end_time_data.append(time_interval)

                        if patience <= 0:
                            print('Stopping training as validation accuracy did not improve '
                                f'for {start_patience} epochs')
                            break

                    logging.info(f'Index: {i}, DataSet: {dataset}, Model: {model_type}, Loss: {loss}, Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}, Epochs: {epoch}')
                    if i == 9:
                        logging.info(f'Epoch time:\n {epoch_end_time_data}')
                    ma_f1_list.append(macro_f1)
                    mi_f1_list.append(micro_f1)
                    
                    show_model(model)
                    
                avg_ma_f1 = np.mean(ma_f1_list)
                avg_mi_f1 = np.mean(mi_f1_list)
                logging.info(f'DataSet: {dataset}, Model: {model_type}, Avg Macro F1: {avg_ma_f1:.4f}, Avg Micro F1: {avg_mi_f1:.4f}')
