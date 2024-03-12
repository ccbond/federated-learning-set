import logging
import torch
from data_loader.hete_graph_data import get_batch_size_list, get_data_target_node_type, load_full_dataset, all_datasets
from fedsat import init_server
from model.init import init_model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def fed_node_classification(model_name: str, data_type: str, federated_type: str):
    datasets = [data_type]
    if data_type == "all":
        datasets = all_datasets
        
        
    for dataset_name in datasets:    
        data = load_full_dataset(dataset_name, True, True, True)
        target_node_type = get_data_target_node_type(dataset_name)    
        num_classes = data[target_node_type].y.max().item() + 1
        batch_size_list = get_batch_size_list(dataset_name)
        
        if model_name == 'han' or model_name == 'hansa' or model_name == 'locsa':
            for batch_size in batch_size_list:
                logging.info(f"Start init server: {model_name} on {dataset_name} with {federated_type}, batch_size: {batch_size}")
                
                macro_f1_list = []
                micro_f1_list = []
                
                for i in range(10):
                    model = init_model(model_name, num_classes, data)
                    server = init_server(federated_type, model, data, target_node_type, batch_size, device)

                    total_time_const, epoch_end_time_data, macro_f1, micro_f1 = server.run()
                    logging.info(f'Index: {i}, Total time cost: {total_time_const}, Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}')

                    if i == 9:
                        logging.info(f'Epoch time:\n {epoch_end_time_data}')
                    macro_f1_list.append(macro_f1)
                    micro_f1_list.append(micro_f1)
                    
                logging.info(f"Dataset: {dataset_name}, model: {federated_type+model_name},Average Macro F1: {sum(macro_f1_list)/10:.4f}, Average Micro F1: {sum(micro_f1_list)/10:.4f}")
