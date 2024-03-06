import copy
import time

from sklearn.metrics import f1_score

from utils import fmodule
import torch
import torch.nn.functional as F

from .fedbase import BasicServer, BasicClient

class FedAvgWithShareAttentionServer(BasicServer):
    def __init__(self, option, model, clients, data = None, target_node_type = None, device = None):
        super(FedAvgWithShareAttentionServer, self).__init__(option, model, clients, data, target_node_type, device)

        self.share_atte = None

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

    def iterate(self):
        model_tensor_dict, model_atte_dict = self.communicate(self.clients)
        agg_model_tensor = self.aggregate(model_tensor_dict)
        agg_model_atte = self.aggregate_attention(model_atte_dict)

        self.set_share_atte(agg_model_atte)

        agg_model = copy.deepcopy(self.model)
        state_dict_keys = agg_model.get_state_dict_keys()

        agg_model = fmodule.model_from_flattened_tensor(agg_model_tensor, agg_model, state_dict_keys)  # Use agg_model instead of ms[0] for clarity
        self.model = agg_model
        return

    def communicate(self, clients):
        c_model_tensor_dict = {}
        c_model_atte_dict = {}
        for idx, c in clients.items():
            model, _loss = c.reply2(self.share_atte)
            model_tensor = fmodule.model_to_tensor(model)
            c_model_tensor_dict[idx] = model_tensor
            c_model_atte_dict[idx] = model.get_now_atte()
        return c_model_tensor_dict, c_model_atte_dict
    
    def communicate_with(self, client_id):
        svr_pkg = self.pack()
        return self.clients[client_id].reply(svr_pkg, self.model_state_dict_keys)
    
    def aggregate_attention(self, model_atte_dict):
        print(model_atte_dict)
        
        agg_model_atte = model_atte_dict[0]
        for idx in range(1, len(model_atte_dict)):
            agg_model_atte += model_atte_dict[idx]
        return agg_model_atte / len(model_atte_dict)
    
    def set_share_atte(self, share_atte):
        self.share_atte = share_atte
        return

class FedAvgWithShareAttentionClient(BasicClient):
    def __init__(self, option, name='', train_data=None, model=None, optimizer=None, device=None):
        super(FedAvgWithShareAttentionClient, self).__init__(option, name, train_data, model, optimizer, device)
        
    def train(self, model, share_atte=None):
        model.train()
        optimizer = self.optimizer
        
        optimizer.zero_grad()
        data_loader = self.train_data.to(self.device)
        out, _ = model(data_loader.x_dict, data_loader.edge_index_dict, self.target_node_type, True, share_atte)
        mask = data_loader[self.target_node_type].train_mask
        loss = F.cross_entropy(out[mask], data_loader[self.target_node_type].y[mask])
        loss.backward()
        optimizer.step()
        
        return loss, model

    def reply2(self, share_atte):
        model = self.model_example.to(self.device)
        loss, model = self.train(model, share_atte)
        return model, loss