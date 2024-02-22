import torch
from data_loader.hete_graph_data import get_data_target_node_type, load_full_dataset
from fedsat import init_server
from model.init import init_model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def fed_node_classification(model_name: str, dataset_name: str, federated_type: str):
    data = load_full_dataset(dataset_name, True, True, True)
    num_classes = data[target_node_type].y.max().item() + 1
    target_node_type = get_data_target_node_type(dataset_name)
    
    if model_name == 'han':
        model = init_model(model_name, num_classes, data)
        server = init_server(federated_type, model, data, target_node_type, device)
        server.run()