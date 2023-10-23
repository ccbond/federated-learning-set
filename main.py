import torch
import logging
from data_loader.hete_graph_data import load_hete_graph_data
from train.hete import train
from model.model import init_model
from utils.args import args


if __name__ == '__main__':
    """
    Main function for training and evaluating heterogeneous model
    """
    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.train_type == 'fed':
        # result = load
        load_fed_hete_graph_data(args.dataset, device)
    else: 
        # load data
        result = load_hete_graph_data(args.dataset, device)
        if result is not None:
            graph, num_classes, train_mask, val_mask, test_mask, y = result

            # load model
            model = init_model(num_classes, graph, device)

            

            # # train model
            # final_best_acc = train(model, graph, y, train_mask,
            #                     val_mask, test_mask, args.epochs, device)
            # print(final_best_acc)
            # # logging.info("Final test accuracy: %.3f" % final_best_acc)
