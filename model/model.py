
from model.han import HAN
from utils.args import args


def init_model(num_classes, device):
    if args.model_name == 'han':
        model = HAN(-1, 64, num_classes).to(device)
        return model
