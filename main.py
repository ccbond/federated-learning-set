import os
import click
import numpy as np
import torch
import logging
import datetime
from model.show import show_model_info
from train.fed_train import fed_node_classification
from train.not_fed_train import no_fed_node_classification, no_fed_node_classification_select_one
from tools.show_and_store_dataset_info import show_and_store_dataset_info

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
def nofed(model, dataset):
    no_fed_node_classification(model, dataset)
    print("No fed run finish.")    

@click.command()
@click.option('--model', default="han", help='The model name')
@click.option('--dataset', default="DBLP", help='The dataset name')
def nofed_s1(model, dataset):
    no_fed_node_classification_select_one(model, dataset)
    print("No fed run finish.")    

@click.command()
@click.option('--model', default="han", help='The model name')
@click.option('--dataset', default="DBLP", help='The dataset name')
@click.option('--fedtype', default="fedavg", help='The federated type of training')
def fed(model, dataset, fedtype):
    fed_node_classification(model, dataset, fedtype)
    print("Fed run finish.")

@click.command()
@click.option('--dataset', default="all", help='The dataset name')
def get_datasets(dataset):
    click.echo("Show and store dataset info")
    show_and_store_dataset_info(dataset)

@click.command()
@click.option('--model', default="han", help='The model name')
@click.option('--dataset', default="IMDB", help='The dataset name')
def show_model(model, dataset):
    click.echo("Show model info")
    show_model_info(model, dataset)

@click.group()
def cli():
    pass    

cli.add_command(get_datasets)
cli.add_command(nofed)
cli.add_command(fed)
cli.add_command(show_model)
cli.add_command(nofed_s1)


if __name__ == '__main__':
    """
    Main function for training and evaluating heterogeneous model
    """
    cli()
