import torch
from data_loader.hete_graph_data import load_hete_graph_data
from heterogeneous.hete import train
from model.model import init_model
from utils.args import args


if __name__ == '__main__':
    ###
    #   Main function for training and evaluating heterogeneous model
    ###

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    result = load_hete_graph_data(args.dataset, device)
    if result is not None:
        graph, num_classes, train_mask, val_mask, test_mask, y = result

        # load model
        model = init_model(num_classes, device)

        # train model
        final_best_acc = train(model, graph, y, train_mask,
                               val_mask, test_mask, args.epochs, device)

    else:
        print("Failed to load graph data.")
