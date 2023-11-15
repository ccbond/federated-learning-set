import torch
import logging
import datetime
from data_loader.hete_graph_data import load_fed_hete_graph_data, load_hete_graph_data
from train.train import train
from model.init import init_model
from utils.args import args


if __name__ == '__main__':
    """
    Main function for training and evaluating heterogeneous model
    """
    now = datetime.datetime.now()
    log_file = now.strftime('%Y-%m-%d') + '.log'
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode = 'w',  format='%(asctime)s - %(message)s')    
    
    logging.info("Arguments: %s", args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Device: %s", device.type)

    if args.train_type == 'fed':
        logging.info("1. Start load fed-heterogeneous-graph data.")

        fed_data = load_fed_hete_graph_data(args.dataset, device)
        
        if fed_data is not None:

            model_list = []
            graph_list = []
            features_list = []
            y_list = []
            train_mask_list = []
            val_mask_list = []
            test_mask_list = []

            # init one model for every subgraph
            for i in range(fed_data):
                graph, num_classes, train_mask, val_mask, test_mask, y = fed_data[i]
                model = init_model(num_classes, graph, device)
                model_list.append(model)
                graph_list.append(graph)
                features_list.append(fe)

            # fed train models
            
            




                

    else:
        logging.info("1. Start load heterogeneous-graph data.")
        data = load_hete_graph_data(args.dataset, device)
        if data is not None:
            graph, num_classes, train_mask, val_mask, test_mask, y = data

            # load model
            model = init_model(num_classes, graph, device)

            # train model
            model, graph, final_best_acc = train(model, graph, y, train_mask,
                                val_mask, test_mask, args.epochs, device)
            print('Final test accuracy: ', final_best_acc)
            logging.info("Final test accuracy: %.3f" % final_best_acc)

    print("Run finish.")
