import copy
import logging
import time

from sklearn.metrics import f1_score

from utils import fmodule
import torch
import torch.nn.functional as F

from .fedbase import BasicServer, BasicClient

class LocalWithShareAttentionServer(BasicServer):
    def __init__(self, option, model, clients, data = None, target_node_type = None, device = None):
        super(LocalWithShareAttentionServer, self).__init__(option, model, clients, data, target_node_type, device)

        self.share_atte = None

    def run(self):        
        # logging.info(f"Total Time Cost")
        total_start_time = time.time()

        start_patience = patience = 200
        best_macro_f1 = 0
        

        epoch_start_time = time.time()
        
        epoch_end_time_data = []
        
        round = 0
        while True:
            round += 1
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
                
            epoch_end_time = time.time()
            origin_time_interval = epoch_end_time - epoch_start_time
            time_interval = "%.3f" % origin_time_interval
            epoch_end_time_data.append(time_interval)
                
            if patience <= 0:
                print('Stopping training as validation accuracy did not improve '
                    f'for {start_patience} epochs')
                break

            # logging.info(f'Epoch: {round:03d}, Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}')
            # logging.info(f"Time cost for round {round}: {iter_end_time - iter_start_time}")
        
        total_end_time = time.time()
        total_time_cost = total_end_time - total_start_time
        return total_time_cost, epoch_end_time_data, macro_f1, micro_f1

    def iterate(self):
        model_tensor_dict, model_atte_dict = self.communicate(self.clients)
        # agg_model_tensor = self.aggregate(model_tensor_dict)
        # print(f'model_atte_dict {model_atte_dict}')
        agg_model_atte = self.aggregate_attention(model_atte_dict)
        # print(f'agg_model_atte {agg_model_atte}')

        self.set_share_atte(agg_model_atte)
        # agg_model = copy.deepcopy(self.model)
        # state_dict_keys = agg_model.get_state_dict_keys()
        # agg_model = fmodule.model_from_flattened_tensor(agg_model_tensor, agg_model, state_dict_keys)  # Use agg_model instead of ms[0] for clarity
        # self.model = agg_model
        # print(self.clients)
        self.model = self.clients['0'].get_model()
        return

    def communicate(self, clients):
        c_model_tensor_dict = {}
        c_model_atte_dict = {}
        for idx, c in clients.items():
            model, _loss = c.reply2(self.share_atte)
            model_tensor = fmodule.model_to_tensor(model)
            c_model_tensor_dict[idx] = model_tensor
            c_model_atte_dict[idx] = fmodule.deep_copy_complex_structure(c.get_now_atte())
        return c_model_tensor_dict, c_model_atte_dict
    
    def communicate_with(self, client_id):
        svr_pkg = self.pack()
        return self.clients[client_id].reply(svr_pkg, self.model_state_dict_keys)
    
    def aggregate_attention(self, data):
        # 初始化，用于存储每个id下字段的张量列表
        aggregate_dict = {}

        # 遍历每个客户端的数据
        for cid, client_data in data.items():
            for id, idata in client_data.items():
                if id not in aggregate_dict:
                    aggregate_dict[id] = {key: [] for key in idata.keys()}  # 初始化键值列表

                for key, value in idata.items():
                    if value is not None:  # 只处理非None值
                        aggregate_dict[id][key].append(value)

        # 对每个id和字段的所有张量求平均，忽略None值
        averaged_tensors = {}
        for id, fields in aggregate_dict.items():
            averaged_tensors[id] = {}
            for key, items in fields.items():
                if items:  # 如果列表不为空，计算非None值的平均
                    stacked_tensors = torch.stack(items)
                    mean_tensor = torch.mean(stacked_tensors, dim=0)
                    averaged_tensors[id][key] = mean_tensor
                else:  # 如果字段在所有客户端中都是None
                    averaged_tensors[id][key] = None

        return averaged_tensors

    
    def set_share_atte(self, share_atte):
        self.share_atte = share_atte
        return
    
    
    def test(self):
        self.model.eval()
        mask = self.data[self.target_node_type].test_mask
        out, _, _ = self.model(self.data.x_dict, self.data.edge_index_dict, self.target_node_type)
        preds = out[mask].argmax(dim=-1)
        labels = self.data[self.target_node_type].y[mask]
        return preds, labels
    

class LocalWithShareAttentionClient(BasicClient):
    def __init__(self, option, name='', train_data=None, model=None, optimizer=None, device=None):
        super(LocalWithShareAttentionClient, self).__init__(option, name, train_data, model, optimizer, device)
        
        self.now_atte = None
        
    def train(self, model, share_atte=None):
        model.train()
        optimizer = self.optimizer
        
        optimizer.zero_grad()
        data_loader = self.train_data.to(self.device)
        out, _, atte  = model(data_loader.x_dict, data_loader.edge_index_dict, self.target_node_type, True, share_atte)
        mask = data_loader[self.target_node_type].train_mask
        loss = F.cross_entropy(out[mask], data_loader[self.target_node_type].y[mask])
        loss.backward(retain_graph=True)
        optimizer.step()
        self.set_now_atte(atte)
        
        return loss, model

    def reply2(self, share_atte):
        model = self.model_example.to(self.device)
        loss, model = self.train(model, share_atte)
        self.set_model(model)
        return model, loss
    
    def set_now_atte(self, now_atte):
        self.now_atte = now_atte
        return
    
    def get_now_atte(self):
        return self.now_atte