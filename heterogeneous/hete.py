import os
import torch
import logging
import bz2
import pickle

from tqdm import tqdm


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


def train(model, graph, y, train_mask, val_mask, test_mask, epochs, device):
    """
    Train the model.

    Parameters:
    - model: The PyTorch model to train.
    - graph: The graph data.
    - y: The ground truth labels.
    - train_mask: The mask for the training set.
    - val_mask: The mask for the validation set.
    - test_mask: The mask for the test set.
    - epochs: The number of epochs to train for.
    - device: The device to run the model on (CPU or GPU).

    Returns:
    - float: The best accuracy achieved on the test set.
    """
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=1e-4)  # Initialize optimizer
    loss_function = torch.nn.CrossEntropyLoss().to(
        device)  # Initialize loss function
    min_epochs = 5
    best_val_acc = 0
    final_best_acc = 0

    for epoch in tqdm(range(epochs)):
        f = model(graph)
        loss = loss_function(f[train_mask], y[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        val_acc, _val_loss = test(model, graph, y, val_mask, test_mask, device)
        test_acc, _test_loss = test(
            model, graph, y, test_mask, test_mask, device)
        if epoch + 1 > min_epochs and val_acc > best_val_acc:
            best_val_acc = val_acc
            final_best_acc = test_acc
        tqdm.write('epoch: {:03d}, train_loss: {:.5f}, val_acc: {:.3f}, test_acc {:.3f}'.format(
            epoch, loss.item(), val_acc, test_acc))
        
        state_dict = model.state_dict()
        compressed = bz2.compress(pickle.dumps(state_dict))
        logging.info('HAN compressed parameters:')
        print(compressed)
        print(type(compressed))
        return
    return final_best_acc
