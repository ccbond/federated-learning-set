import logging
from model.han import HAN


def init_model(model, num_classes, graph, device):
    """
    Initialize the model based on the specified arguments.

    Parameters:
    - num_classes (int): The number of target classes.
    - graph: The graph data.
    - device: The device to run the model on (CPU or GPU).

    Returns:
    - model: The initialized PyTorch model, or None if the model is not found.
    """
    if model == 'han':
        model = HAN(-1, 64, num_classes, graph).to(device)
        return model
    else:
        logging.info("Model not found.")
        return None
