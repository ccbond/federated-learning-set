import os.path as osp
from data_loader.hete_graph_data import all_datasets, load_full_dataset

def save_dataset_info(data_name: str, data: any, path: str):
    with open(path, 'a') as f:
        f.write(f"Dataset: {data_name}\n")
        f.write(str(data) + '\n')
        f.write('\n')

def show_and_store_dataset_info(dataset: str = "all"):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../result/datasets_info.txt')
    if dataset == "all":
        for dataset in all_datasets:
            save_dataset_info(dataset, load_full_dataset(dataset, False, False), path)
            print(f"Dataset: {dataset} is written.")
    else:
        save_dataset_info(dataset, load_full_dataset(dataset, False, False), path)
        print(f"Dataset: {dataset} is written.")
