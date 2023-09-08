import torch
from data_loader.hete_graph_data import load_hete_graph_data
from heterogeneous.hete import train
from model.model import init_model
from utils.args import args_parser

args = args_parser()


if __name__ == '__main__':
    ###
    #   Main function for training and evaluating heterogeneous model
    ###
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    graph, num_classes, train_mask, val_mask, test_mask, y = load_hete_graph_data(
        args.dataset, device)

    # load model
    model = init_model()

    # train model
    final_best_acc = train(model, graph, y, train_mask,
                           val_mask, test_mask, args.epochs, device)
