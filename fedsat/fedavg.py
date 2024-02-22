from .fedbase import BasicServer, BasicClient

class FedAvgServer(BasicServer):
    def __init__(self, option, model, clients, test_data = None, device = None):
        super(FedAvgServer, self).__init__(option, model, clients, test_data, device)

class FedAvgClient(BasicClient):
    def __init__(self, option, name='', train_data=None, device=None):
        super(FedAvgClient, self).__init__(option, name, train_data, device)
