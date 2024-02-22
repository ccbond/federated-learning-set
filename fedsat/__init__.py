from fedsat.fedavg import FedAvgClient, FedAvgServer
from torch_geometric.loader import NeighborLoader
import torch
import torch.nn.functional as F


def init_server(fed_method, model, data, target_node_type, device): 
    train_idx = data[target_node_type].train_mask.nonzero(as_tuple=True)[0]
    train_loader = NeighborLoader(data, num_neighbors=[32]*3, input_nodes=(target_node_type, train_idx), batch_size=128, shuffle=False)

    data_list = []
    for data in train_loader:
        data_list.append(data)
    num_clients = len(data_list)

    lr = 0.02
    weight_decay = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    client_option = {'batch_size': 128, 'learning_rate': lr, 'weight_decay': weight_decay, 'num_steps': 10, 'epochs': 100, 'target_node_type': target_node_type}

    if fed_method == 'fedavg':
        clients = {}
        for i in range(num_clients):
            idx = str(i)
            
            client = FedAvgClient(option=client_option, name=idx, train_data=data_list[i], model=model, optimizer=optimizer, device=device)
            clients[idx] = client
        
        server_option = {'num_rounds': 100, 'learning_rate': 0.001}
        server = FedAvgServer(option=server_option, model=model, clients=clients, data=data, device=device)
        return server
    
    else:
        return None
