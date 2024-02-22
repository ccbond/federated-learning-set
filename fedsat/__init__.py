from fedsat.fedavg import FedAvgClient, FedAvgServer
from torch_geometric.loader import NeighborLoader


def init_server(num_clients, fed_method, model, data, target_node_type, device): 
    train_idx = data[target_node_type].train_mask.nonzero(as_tuple=True)[0]
    train_loader = NeighborLoader(data, num_neighbors=[32]*3, input_nodes=(target_node_type, train_idx), batch_size=128, shuffle=False)

    if fed_method == 'fedavg':
        clients = []
        for i in range(num_clients):
            idx = str(i)
            client_option = {'batch_size': 128, 'learning_rate': 0.001, 'weight_decay': 0.0005, 'num_steps': 10, 'epochs': 100, 'target_node_type': target_node_type}

            client = FedAvgClient(option=client_option, name=idx, train_data=train_loader[i], model=model, device=device)
            clients.append(client)
        
        server_option = {'num_rounds': 100, 'learning_rate': 0.001}
        server = FedAvgServer(option=server_option, model=model, clients=clients, data=data, device=device)
        return server
    
    else:
        return None