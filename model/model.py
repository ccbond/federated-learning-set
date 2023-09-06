
import logging
from model.han import HAN
from utils.args import args


def init_model(num_classes, graph, device):
    if args.model == 'han':
        model = HAN(-1, 64, num_classes, graph).to(device)
        return model
    else:
        logging.info("Model not found.")
        return None
