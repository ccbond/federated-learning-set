import collections
import copy
import logging
import time

from sklearn.metrics import f1_score

from utils import fmodule
import torch
import torch.nn.functional as F


class BasicServer:
    def __init__(self, option, model, clients, data, target_node_type, device):
        self.device = device
        self.data = data.to(self.device)
        self.target_node_type = target_node_type
        self.model = model.to(self.device)
        self.model_state_dict_keys = model.get_state_dict_keys()
        self.clients = clients
        self.num_clients = len(clients)
        self.decay_rate= option['learning_rate']
        self.current_round = -1
        self.option = option
        self.set_client_server()
        
    def __str__(self) -> str:
            return (
                f"BasicServer(\n"
                f"  Device: {self.device},\n"
                f"  Target Node Type: {self.target_node_type},\n"
                f"  Number of Clients: {self.num_clients},\n"
                f"  Number of Rounds: {self.num_rounds},\n"
                f"  Current Round: {self.current_round},\n"
                f")"
            ) 
 
    def run(self):        
        # logging.info(f"Total Time Cost")
        total_start_time = time.time()

        start_patience = patience = 100
        best_macro_f1 = 0
        
        while True:
            self.current_round = round
            # iter_start_time = time.time()
            # federated train
            self.iterate()
            # decay learning rate
            # self.global_lr_scheduler(round)
            # update client model
            self.update_clients()
            
            # iter_end_time = time.time()
            preds, labels = self.test()

            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()

            macro_f1 = f1_score(labels, preds, average='macro')
            micro_f1 = f1_score(labels, preds, average='micro')
            
            if best_macro_f1 <= macro_f1:
                patience = start_patience
                best_macro_f1 = macro_f1
            else: 
                patience -= 1
                
            if patience <= 0:
                print('Stopping training as validation accuracy did not improve '
                    f'for {start_patience} epochs')
                break

            # logging.info(f'Epoch: {round:03d}, Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}')
            # logging.info(f"Time cost for round {round}: {iter_end_time - iter_start_time}")
        
        total_end_time = time.time()
        total_time_cost = total_end_time - total_start_time
        return total_time_cost, macro_f1, micro_f1

    def get_agg_model_tensor(self):
        return fmodule.model_to_tensor(self.model)

    def iterate(self):
        model_tensor_dict = self.communicate(self.clients)
        agg_model_tensor = self.aggregate(model_tensor_dict)

        agg_model = copy.deepcopy(self.model)
        state_dict_keys = agg_model.get_state_dict_keys()

        agg_model = fmodule.model_from_flattened_tensor(agg_model_tensor, agg_model, state_dict_keys)  # Use agg_model instead of ms[0] for clarity
        self.model = agg_model
        return

    def communicate(self, clients):
        c_model_tensor_dict = {}
        for idx, c in clients.items():
            model, _loss = c.reply2()
            model_tensor = fmodule.model_to_tensor(model)
            c_model_tensor_dict[idx] = model_tensor
        return c_model_tensor_dict
    
    def communicate_with(self, client_id):
        svr_pkg = self.pack()
        return self.clients[client_id].reply(svr_pkg, self.model_state_dict_keys)
            
    def pack(self):
        return {
            "model" : copy.deepcopy(self.model),
        }

    def unpack(self, packages_received_from_clients):
        res = collections.defaultdict(list)
        for cpkg in packages_received_from_clients:
            for pname, pval in cpkg.items():
                res[pname].append(pval)
        return res

    def global_lr_scheduler(self, current_round):
        self.lr = self.option['learning_rate']*1.0/(current_round+1)
        for c in self.clients.values():
            c.set_learning_rate(self.lr)
                
    def aggregate(self, model_state_dict: dict):
        return fmodule.model_mean(model_state_dict)
        
    def set_client_server(self):
        for c in self.clients.values():
            c.set_server(self)

    def test(self):
        self.model.eval()
        mask = self.data[self.target_node_type].test_mask
        out, _, _ = self.model(self.data.x_dict, self.data.edge_index_dict, self.target_node_type)
        preds = out[mask].argmax(dim=-1)
        labels = self.data[self.target_node_type].y[mask]
        return preds, labels
    
    def update_clients(self):
        for idx, c in self.clients.items():
            c.set_model(self.model)
            c.optimizer = torch.optim.Adam(c.model_example.parameters(), lr=c.learning_rate, weight_decay=c.weight_decay)


class BasicClient():
    def __init__(self, option, name='', train_data=None, model=None, optimizer=None, device=None):
        self.device = device
        self.name = name
        # create local dataset
        self.train_data = train_data.to(self.device)
        self.model_example = model
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
        # server
        self.server = None
        self.optimizer = optimizer

    def train(self, model):
        # print('train')
        model.train()
        optimizer = self.optimizer
        
        optimizer.zero_grad()
        data_loader = self.train_data.to(self.device)
        out, _ = model(data_loader.x_dict, data_loader.edge_index_dict, self.target_node_type)
        mask = data_loader[self.target_node_type].train_mask
        loss = F.cross_entropy(out[mask], data_loader[self.target_node_type].y[mask])
        loss.backward()
        optimizer.step()
        
        return loss, model
    def test(self, model):
        model.eval()
        data = self.data_loader.to(self.device)
        
        out, _ = model(data.x_dict, data.edge_index_dict, self.target_node_type)
        preds = out.argmax(dim=-1)
        labels = data[self.target_node_type].y

        return preds, labels

    def reply2(self):
        model = self.model_example.to(self.device)
        loss, model = self.train(model)
        return model, loss

    def train_loss(self, model):
        return self.test(model,'train')['loss']

    def valid_loss(self, model):
        return self.test(model)[1]['loss']

    def set_server(self, server=None):
        if server is not None:
            self.server = server

    def set_learning_rate(self, lr = None):
        self.learning_rate = lr if lr else self.learning_rate

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model_example.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def set_model(self, model):
        self.model_example = model
