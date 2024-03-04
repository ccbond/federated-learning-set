import os.path as osp
from data_loader.hete_graph_data import all_datasets, get_data_target_node_type, load_full_dataset

def save_dataset_info(data_name: str, data: any, path: str):
    target_node_type = get_data_target_node_type(data_name)
    num_classes = data[target_node_type].y.max().item() + 1
    num_train = data[target_node_type].train_mask.sum().item()
    # num_val = data[target_node_type].val_mask.sum().item()
    num_test = data[target_node_type].test_mask.sum().item()

    # y = 

    print(f'y: {data[target_node_type].y}')
    max_y = data[target_node_type].y.max().item()
    max_sum_y = data[target_node_type].y.sum(dim=1).max().item()
    print(f'max_y: {max_y}')
    print(f'max_sum_y: {max_sum_y}')

    with open(path, 'a') as f:
        f.write(f"Dataset: {data_name}\n")
        f.write(str(data) + '\n')
        f.write(f"Target node type: {target_node_type}\n")
        f.write(f"Number of classes: {num_classes}\n")
        f.write(f"Number of training nodes: {num_train}\n")
        # f.write(f"Number of validation nodes: {num_val}\n")
        f.write(f"Number of testing nodes: {num_test}\n")
        f.write('\n')
        f.write('\n')
        f.write('\n')

def show_and_store_dataset_info(dataset: str = "all"):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../result/datasets_info.txt')
    if dataset == "all":
        for dataset in all_datasets:
            data = load_full_dataset(dataset, False, False)
            save_dataset_info(dataset, data, path)
            print(f"Dataset: {dataset} is written.")
    else:
        save_dataset_info(dataset, load_full_dataset(dataset, False, False), path)
        print(f"Dataset: {dataset} is written.")
