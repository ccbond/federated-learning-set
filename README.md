# federated-learning-set
A set about fedreated learning tutriols.

## Start
pip install pyg-lib==0.3 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

pyg-lib 0.4.0 最新修改了neighobr sampler的接口，导致代码无法运行，所以需要降级安装0.3.0版本

## Support methods
1. No-fed heterogeneous graph neural network learning
 - [x] HAN
2. Federated learning of heterogeneous graph neural network
 - [x] FedHAN


## Tools

1. show and store dataset info in a text file
```
python3 tools/show_and_store_dataset_info.py
```

## Result

```
2024-02-18 01:40:30,395 - Device: cuda
2024-02-18 01:40:30,397 - Loading dataset: DBLP
2024-02-18 01:40:50,171 - DataSet: DBLP, Model: han, Loss: 0.19931337237358093, Macro F1: 0.7499, Micro F1: 0.7565, Epochs: 199
```
