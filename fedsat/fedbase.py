import collections
import copy
import logging
import time

from utils import fmodule
import torch
import torch.nn.functional as F


class BasicServer:
    def __init__(self, option, model, clients, data, device):
        self.device = device
        self.data = data.to(self.device)
        self.model = model.to(self.device)
        self.clients = clients
        self.num_clients = len(clients)
        self.num_rounds = option['num_rounds']
        self.decay_rate= option['learning_rate_decay']
        self.current_round = -1
        self.option = option
        self.set_client_server()
 
    def run(self):
        logging.info(f"Total Time Cost")
        total_start_time = time.time()
        for round in range(self.num_rounds + 1):
            self.current_round = round
            logging.info(f"-----------------Round {round}---------------")
            iter_start_time = time.time()
            # federated train
            self.iterate()
            # decay learning rate
            self.global_lr_scheduler(round)
            iter_end_time = time.time()
            logging.info(f"Time cost for round {round}: {iter_end_time - iter_start_time}")
        
        total_end_time = time.time()
        logging.info("=================END================")
        logging.info('Total time cost: {}'.format(total_end_time - total_start_time))
        return
            
    def iterate(self):
        # training
        models = self.communicate(self.clients)['model']
        # aggregate
        self.model = self.aggregate(models, p=[1.0 * 1 / self.num_clients])
        return
            
    def communicate(self, clients):
        packages_received_from_clients = []
        client_package_buffer = {}
        
        communicate_clients = list(set(clients))
        for cid in communicate_clients:client_package_buffer[cid] = None
        for client_id in communicate_clients:
            response_from_client_id = self.communicate_with(client_id)
            packages_received_from_clients.append(response_from_client_id)
        for i, cid in enumerate(communicate_clients):
            client_package_buffer[cid] = packages_received_from_clients[i]
        packages_received_from_clients = [client_package_buffer[cid] for cid in clients]
        return self.unpack(packages_received_from_clients)
    
    def communicate_with(self, client_id):
        svr_pkg = self.pack(client_id)
        return self.clients[client_id].reply(svr_pkg)
            
    def pack(self, client_id):
        """
        Pack the necessary information for the client's local training.
        Any operations of compression or encryption should be done here.
        :param
            client_id: the id of the client to communicate with
        :return
            a dict that only contains the global model as default.
        """
        return {
            "model" : copy.deepcopy(self.model),
        }

    def unpack(self, packages_received_from_clients):
        """
        Unpack the information from the received packages. Return models and losses as default.
        :param
            packages_received_from_clients:
        :return:
            res: collections.defaultdict that contains several lists of the clients' reply
        """
        res = collections.defaultdict(list)
        for cpkg in packages_received_from_clients:
            for pname, pval in cpkg.items():
                res[pname].append(pval)
        return res

    def global_lr_scheduler(self, current_round):
        """
        Control the step size (i.e. learning rate) of local training
        :param
            current_round: the current communication round
        """
        if self.lr_scheduler_type == -1:
            return
        elif self.lr_scheduler_type == 0:
            """eta_{round+1} = DecayRate * eta_{round}"""
            self.lr*=self.decay_rate
            for c in self.clients:
                c.set_learning_rate(self.lr)
        elif self.lr_scheduler_type == 1:
            """eta_{round+1} = eta_0/(round+1)"""
            self.lr = self.option['learning_rate']*1.0/(current_round+1)
            for c in self.clients:
                c.set_learning_rate(self.lr)
                
    def aggregate(self, models: list, p=[]):
        """
        Aggregate the locally improved models.
        :param
            models: a list of local models
            p: a list of weights for aggregating
        :return
            the averaged result

        pk = nk/n where n=self.data_vol
        K = |S_t|
        N = |S|
        -------------------------------------------------------------------------------------------------------------------------
         weighted_scale                 |uniform (default)          |weighted_com (original fedavg)   |other
        ==========================================================================================================================
        N/K * Σpk * model_k             |1/K * Σmodel_k             |(1-Σpk) * w_old + Σpk * model_k  |Σ(pk/Σpk) * model_k
        """
        if len(models) == 0: return self.model
        if self.aggregation_option == 'weighted_scale':
            K = len(models)
            N = self.num_clients
            return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)]) * N / K
        elif self.aggregation_option == 'uniform':
            return fmodule._model_average(models)
        elif self.aggregation_option == 'weighted_com':
            w = fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])
            return (1.0-sum(p))*self.model + w
        else:
            sump = sum(p)
            p = [pk/sump for pk in p]
            return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])
        
    def set_client_server(self):
        for c in self.clients:
            c.set_server(self)

class BasicClient():
    def __init__(self, option, name='', train_data=None, model=None, device=None):
        self.device = device
        self.name = name
        # create local dataset
        self.train_data = train_data.to(self.device)
        self.datavol = len(self.train_data)
        self.data_loader = None
        # local calculator
        # hyper-parameters for training
        self.batch_size = option['batch_size']
        self.learning_rate = option['learning_rate']
        self.weight_decay = option['weight_decay']
        self.num_steps = option['num_steps']
        self.epochs = option['epochs']
        self.target_node_type = option['target_node_type']
        self.model = None
        # server
        self.server = None
        self.target_node_type = None
        self.optimizer = None

    @fmodule.with_multi_gpus
    def train(self, model):
        model.train()
        optimizer = self.optimizer
        
        optimizer.zero_grad()
        data_loader = self.data_loader.to(self.device)
        out, _ = model(data_loader.x_dict, data_loader.edge_index_dict, self.target_node_type)
        mask = data_loader[self.target_node_type].train_mask
        loss = F.cross_entropy(out[mask], data_loader[self.target_node_type].y[mask])
        loss.backward()
        optimizer.step()
        
        return loss

    @ fmodule.with_multi_gpus
    def test(self, model):
        model.eval()
        data = self.data_loader.to(self.device)

        
        out, _ = model(data.x_dict, data.edge_index_dict, self.target_node_type)
        preds = out.argmax(dim=-1)
        labels = data[self.target_node_type].y

        return preds, labels

    def unpack(self, received_pkg):
        """
        Unpack the package received from the server
        :param
            received_pkg: a dict contains the global model as default
        :return:
            the unpacked information that can be rewritten
        """
        # unpack the received package
        return received_pkg['model']

    def reply(self, svr_pkg):
        """
        Reply to server with the transmitted package.
        The whole local procedure should be planned here.
        The standard form consists of three procedure:
        unpacking the server_package to obtain the global model,
        training the global model, and finally packing the updated
        model into client_package.
        :param
            svr_pkg: the package received from the server
        :return:
            client_pkg: the package to be send to the server
        """
        model = self.unpack(svr_pkg)
        self.train(model)
        cpkg = self.pack(model)
        return cpkg

    def pack(self, model):
        """
        Packing the package to be send to the server. The operations of compression
        of encryption of the package should be done here.
        :param
            model: the locally trained model
        :return
            package: a dict that contains the necessary information for the server
        """
        return {
            "model" : model,
        }


    def train_loss(self, model):
        """
        Get the task specified loss of the model on local training data
        :param model:
        :return:
        """
        return self.test(model,'train')['loss']

    def valid_loss(self, model):
        """
        Get the task specified loss of the model on local validating data
        :param model:
        :return:
        """
        return self.test(model)[1]['loss']

    def set_server(self, server=None):
        if server is not None:
            self.server = server

    def set_learning_rate(self, lr = None):
        """
        set the learning rate of local training
        :param lr:
        :return:
        """
        self.learning_rate = lr if lr else self.learning_rate

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
