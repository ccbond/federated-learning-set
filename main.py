import os
import click
import torch
import logging
import datetime
from notfed import no_fed_node_classification
from tools.show_and_store_dataset_info import show_and_store_dataset_info
from train.train import train
from model.init import init_model


# Set log config
now = datetime.datetime.now()

logs_dir = 'logs'
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

log_file = os.path.join(logs_dir, now.strftime('%Y-%m-%d') + '.log')
logging.basicConfig(level=logging.INFO, filename=log_file,
                    filemode='w',  format='%(asctime)s - %(message)s')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info("Device: %s", device.type)

@click.command()
@click.option('--model', default="han", help='The model name')
@click.option('--dataset', default="DBLP", help='The dataset name')
@click.option('--epochs', default=100, help='The rounds of training')
def nofed(model, dataset, epochs):
    no_fed_node_classification(model, dataset, epochs)
    print("No fed run finish.")    


@click.command()
@click.option('--model', default="han", help='The model name')
@click.option('--dataset', default="DBLP", help='The dataset name')
@click.option('--epochs', default=100, help='The rounds of training')
@click.option('--train_type', default="fed", help='The type of training')
@click.option('--client_nums', default=10, help='The number of clients')
def fed(model, dataset, epochs, train_type, client_nums):
    print("Fed run finish.")

@click.command()
@click.option('--dataset', default="all", help='The dataset name')
def get_datasets(dataset):
    click.echo("Show and store dataset info")
    show_and_store_dataset_info(dataset)

@click.group()
def cli():
    pass    

cli.add_command(get_datasets)
cli.add_command(nofed)


if __name__ == '__main__':
    """
    Main function for training and evaluating heterogeneous model
    """
    cli()

