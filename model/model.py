
import logging
from model.han import HAN
from utils.args import args


def init_model(num_classes, graph, device):
    """
    Initialize the model based on the specified arguments.

    Parameters:
    - num_classes (int): The number of target classes.
    - graph: The graph data.
    - device: The device to run the model on (CPU or GPU).

    Returns:
    - model: The initialized PyTorch model, or None if the model is not found.
    """
    if args.model == 'han':
        model = HAN(-1, 64, num_classes, graph).to(device)
        # some model parameters will init in the frist forward.
        # params_list = model.parameters()
        # logging.info('HAN parameters: ', [p.shape for p in params_list]）
        # model.conv1
        return model
    else:
        logging.info("Model not found.")
        return None
