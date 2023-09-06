import os
import torch
import logging
from tqdm import tqdm


@torch.no_grad()
def test(model, graph, y, mask, test_mask, device):
    model.eval()
    with torch.no_grad():
        out = model(graph)
        loss_function = torch.nn.CrossEntropyLoss().to(device)
        loss = loss_function(out[mask], y[mask])

    _, pred = out.max(dim=1)
    correct = int(pred[mask].eq(y[mask]).sum().item())
    acc = correct / int(test_mask.sum())

    return acc, loss.item()


def train(model, graph, y, train_mask, val_mask, test_mask, epochs, device):
    logging.info("model is %s" % model)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
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

    return final_best_acc
