from typing import Callable, List, Optional
import os.path as osp
import numpy as np
from torch_geometric.data import HeteroData, InMemoryDataset
from scipy import io as sio
import torch
from torch_geometric.transforms import AddMetaPaths


    
def load_acm_data():
        data = HeteroData()
        mat_path = osp.join(osp.dirname(osp.realpath(__file__)), '../data/ACM/ACM.mat')

        mat_data = sio.loadmat(mat_path)

        p_vs_l = mat_data['PvsL']
        p_vs_a = mat_data['PvsA']       # paper-author
        p_vs_t = mat_data['PvsT']       # paper-term, bag of words
        p_vs_c = mat_data['PvsC']       # paper-conference, labels come from that
        
        conf_ids = [0, 1, 9, 10, 13]
        label_ids = [0, 1, 2, 2, 1]

        p_vs_c_filter = p_vs_c[:, conf_ids]
        p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]

        p_vs_l = p_vs_l[p_selected]
        p_vs_a = p_vs_a[p_selected]
        p_vs_t = p_vs_t[p_selected]
        p_vs_c = p_vs_c[p_selected]
        
        pc_p, pc_c = p_vs_c.nonzero()
        labels = np.zeros(len(p_selected), dtype=np.int64)
        for conf_id, label_id in zip(conf_ids, label_ids):
            labels[pc_p[pc_c == conf_id]] = label_id
        labels = torch.LongTensor(labels)


        data['paper'].x = torch.FloatTensor(p_vs_t.toarray())
        data['paper'].y = labels
        
        
        data['paper', 'writes', 'author'].edge_index = torch.tensor(np.vstack(p_vs_a.nonzero()), dtype=torch.long)
        data['author', 'written_by', 'paper'].edge_index = torch.tensor(np.vstack(p_vs_a.transpose().nonzero()), dtype=torch.long)
        data['paper', 'has_subject', 'subject'].edge_index = torch.tensor(np.vstack(p_vs_l.nonzero()), dtype=torch.long)
        data['subject', 'subject_of', 'paper'].edge_index = torch.tensor(np.vstack(p_vs_l.transpose().nonzero()), dtype=torch.long)

        float_mask = np.zeros(len(pc_p))
        for conf_id in conf_ids:
            pc_c_mask = (pc_c == conf_id)
            float_mask[pc_c_mask] = np.random.permutation(np.linspace(0, 1, pc_c_mask.sum()))
        train_idx = np.where(float_mask <= 0.2)[0]
        val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
        test_idx = np.where(float_mask > 0.3)[0]

        num_nodes = data['paper'].num_nodes
        train_mask = get_binary_mask(num_nodes, train_idx)
        val_mask = get_binary_mask(num_nodes, val_idx)
        test_mask = get_binary_mask(num_nodes, test_idx)
        data['paper'].train_mask = train_mask
        data['paper'].val_mask = val_mask
        data['paper'].test_mask = test_mask
        
        metapaths = [[('paper', 'author'), ('author', 'paper')],[('paper', 'subject'), ('subject', 'paper')]]
        data = AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=True, add_zero_features=True, drop_unconnected_node_types=True)(data)

        return data



def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size, dtype=torch.bool)
    mask[indices] = True
    # transform to bool
    
    return mask