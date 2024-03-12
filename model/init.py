import logging
from model.han import HAN
from model.hansa import HANSA
from model.hgt import HGT


def init_model(model_type, num_classes, graph):
    """
    Initialize the model based on the specified arguments.

    Parameters:
    - num_classes (int): The number of target classes.
    - graph: The graph data.
    - device: The device to run the model on (CPU or GPU).

    Returns:
    - model: The initialized PyTorch model, or None if the model is not found.
    """
    if model_type == 'han':
        model = HAN(-1, num_classes, metadata=graph.metadata())
        return model
    elif model_type == 'hansa' or model_type == 'locsa':
        model = HANSA(-1, 512, metadata=graph.metadata())
        return model
    elif model_type == 'hgt':
        model = HGT(num_classes, metadata=graph.metadata(), data=graph)
        return model 
    else:
        logging.info("Model not found.")
        return None


def all_models():
    return ['han']
