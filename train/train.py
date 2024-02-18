from tqdm import tqdm
from typing import List
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader

def full_train_nc(model, data, optimizer, target_node_type, device) -> float:
    model.train()

    optimizer.zero_grad()        
    
    data = data.to(device)
    out, _ = model(data.x_dict, data.edge_index_dict, target_node_type)
    mask = data[target_node_type].train_mask
    loss = F.cross_entropy(out[mask], data[target_node_type].y[mask])
    loss.backward()
    optimizer.step()
    
    return loss.item()

def mini_batch_train_nc(model, train_loader, optimizer, target_node_type, device) -> float:
    model.train()

    total_examples = 0
    total_loss = 0

    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        out, processed = model(batch.x_dict, batch.edge_index_dict, target_node_type)
        if not processed:
            continue
        
        mask = batch[target_node_type].train_mask
        loss = F.cross_entropy(out[mask], batch[target_node_type].y[mask])
        loss.backward()
        optimizer.step()

        batch_size = mask.sum().item()
        total_examples += batch_size
        total_loss += float(loss) * batch_size

    return total_loss / total_examples

@torch.no_grad()
def full_test_nc(model, data, target_node_type):
    model.eval()

    out, _ = model(data.x_dict, data.edge_index_dict, target_node_type)
    preds = out.argmax(dim=-1)
    labels = data[target_node_type].y

    return preds, labels

@torch.no_grad()
def mini_batch_test_nc(model, test_loader, target_node_type, device):
    model.eval()

    all_preds = []
    all_labels = []

    for batch in test_loader:
        batch = batch.to(device)
        out, processed = model(batch.x_dict, batch.edge_index_dict, target_node_type)
        if not processed:
            continue
        
        preds = out.argmax(dim=-1)  # 将预测移动到CPU并转换为numpy数组

        # 由于批处理中可能存在不同的mask，我们需要单独处理
        # 这里我们收集所有的预测和标签，不区分train/val/test
        labels = batch[target_node_type].y  # 同样将标签移动到CPU并转换为numpy数组

        all_preds.extend(preds)
        all_labels.extend(labels)

    return torch.stack(all_preds), torch.stack(all_labels)


# not federated training for node classification
def no_fed_train_nc(model, data, optimizer, target_node_type, is_mini_batch, device) -> float:
    if is_mini_batch:
        train_idx = data[target_node_type].train_mask.nonzero(as_tuple=True)[0]
        train_loader = NeighborLoader(data, num_neighbors=[32]*3, input_nodes=(target_node_type, train_idx), batch_size=128, shuffle=False)
        
        return mini_batch_train_nc(model, train_loader, optimizer, target_node_type, device)
    else:
        return full_train_nc(model, data, optimizer, target_node_type, device)

# not federated testing for node classification
def no_fed_test_nc(model, data, target_node_type, is_mini_batch, device) -> List[float]:
    if is_mini_batch:
        test_idx = data[target_node_type].test_mask.nonzero(as_tuple=True)[0]
        test_loader = NeighborLoader(data, num_neighbors=[32]*3, input_nodes=(target_node_type, test_idx), batch_size=128, shuffle=False)
        return mini_batch_test_nc(model, test_loader, target_node_type, device)
    else:
        return full_test_nc(model, data, target_node_type)
