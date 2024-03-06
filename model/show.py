import logging

from model.init import all_models, init_model
from data_loader.hete_graph_data import all_datasets, get_data_target_node_type, load_full_dataset

def show_model_info(model_name: str, dataset_name: str):
    if dataset_name == 'all':
        datasets = all_datasets
    else:
        datasets = [dataset_name]
    
    if model_name == 'all':
        models = all_models
    else:
        models = [model_name]

    for model_type in models:
        for dataset in datasets:
            logging.info(f"Loading dataset: {dataset}, model: {model_type}")
            data = load_full_dataset(dataset, True, True, True)
            target_node_type = get_data_target_node_type(dataset)
            num_classes = data[target_node_type].y.max().item() + 1

            model = init_model(model_type, num_classes, data)
            model.train()
            
            print("model state_dict:")
            
            # for param_tensor in model.state_dict():
            #     print(param_tensor, "\t", model.state_dict()[param_tensor])


            for name, param in model.named_parameters():
                if param.is_leaf and param.requires_grad:
                    # 参数已初始化并准备好训练
                    print(name, "=>", param.shape)
                else:
                    # 参数未初始化
                    print(name, "=> Uninitialized")
                    # print(key, "=>", model.state_dict()[key].shape)
                    
            print("model parameters:")
            for param in model.parameters():
                print(param)

            print("model state dict list:")
            print(model.get_state_dict_keys())


def show_model(model):
    for name, param in model.named_parameters():
        if param.is_leaf and param.requires_grad:
            # 参数已初始化并准备好训练
            print(name, "=>", param.shape)
        else:
            # 参数未初始化
            print(name, "=> Uninitialized")
            # print(key, "=>", model.state_dict()[key].shape)
            
    print("model parameters:")
    for param in model.parameters():
        print(param)

    print("model state dict list:")
    print(model.get_state_dict_keys())
