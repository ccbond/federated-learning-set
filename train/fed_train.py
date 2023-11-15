import torch
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().numpy()
    labels = labels.numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_fl = f1_score(labels, prediction, average="micro")
    macro_f1 = f1_score(labels, prediction, average="macro")

    return accuracy, micro_fl, macro_f1

@torch.no_grad()  # Decorator to disable gradient calculation during inference
def test(model, graph, y, mask, test_mask, device):
    """
    Evaluate the model on a test set.

    Parameters:
    - model: The PyTorch model to evaluate.
    - graph: The graph data.
    - y: The ground truth labels.
    - mask: The mask to apply on the labels.
    - test_mask: The mask for the test set.
    - device: The device to run the model on (CPU or GPU).

    Returns:
    - float: The accuracy of the model on the test set.
    - float: The loss on the test set.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Context manager to disable gradient calculation
        out = model(graph)
        loss_function = torch.nn.CrossEntropyLoss().to(device)
        loss = loss_function(out[mask], y[mask])

    _, pred = out.max(dim=1)
    correct = int(pred[mask].eq(y[mask]).sum().item())
    acc = correct / int(test_mask.sum())

    return acc, loss.item()


def train(model, graph, y, features, average_weight, loss_function, train_mask, val_mask, test_mask, epochs, device):
    """
    Train the model.
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    min_epochs = 5
    best_val_acc = 0
    final_best_acc = 0

    
    for epoch in tqdm(range(epochs)):
        f = model(graph, features)
        if epoch == 1:
            f = model(graph, features, average_weight)
            
        loss = loss_function(f[train_mask], y[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        val_acc, _val_loss = test(model, graph, y, val_mask, test_mask, device)
        test_acc, _test_loss = test(model, graph, y, test_mask, test_mask, device)
        if epoch + 1 > min_epochs and val_acc > best_val_acc:
            best_val_acc = val_acc
            final_best_acc = test_acc
        tqdm.write('epoch: {:03d}, train_loss: {:.5f}, val_acc: {:.3f}, test_acc {:.3f}'.format(
            epoch, loss.item(), val_acc, test_acc))
    

    single_weights = model.get_semantic_attention_weights(graph, features)

    return single_weights, final_best_acc


def fed_train(model_list, graph_list, features_list, y_list, train_mask_list, val_mask_list, test_mask_list, epochs, federated_times, device):
    loss_function = torch.nn.CrossEntropyLoss().to(device)

    average_weight = None

    for i in range(federated_times):

        weights = []
        for x in range(len(model_list)):
            curr_model = model_list[x]
            curr_graph = graph_list[x]
            curr_y = y_list[x]
            curr_features = features_list[x]
            curr_train_mask = train_mask_list[x]
            curr_val_mask = val_mask_list[x]
            curr_test_mask = test_mask_list[x]

            curr_weight, curr_best_acc = train(curr_model, curr_graph, curr_y, curr_features, average_weight, loss_function, curr_train_mask, curr_val_mask, curr_test_mask, epochs, federated_times, device)
            weights.append(curr_weight)

            tqdm.write('federated time: {:03d}, model index: {:.5f}, curr_best_accurcy'.format(i, x, curr_best_acc))

        weights_array = np.array(weights)
        average_weight = np.mean(weights_array)
