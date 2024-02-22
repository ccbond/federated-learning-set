from .fedbase import BasicServer, BasicClient

class FedAvgServer(BasicServer):
    def __init__(self, option, model, clients, data = None, target_node_type = None, device = None):
        super(FedAvgServer, self).__init__(option, model, clients, data, target_node_type, device)

class FedAvgClient(BasicClient):
    def __init__(self, option, name='', train_data=None, model=None, optimizer=None, device=None):
        super(FedAvgClient, self).__init__(option, name, train_data, model, optimizer, device)
