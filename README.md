# SS-HAN

This Repository is modified according to the HAN model, the original copy in https://github.com/Jhy1993/HAN, DGL copy in https://github.com/dmlc/dgl/tree/master/examples/pytorch/han, We use a two-stage self-supervised learning method to complete the task of heterogeneous graph networks and named the new model as SS-HAN, The repository is still being updated

# Requirements
- python >= 3.6
- [pytorch >=1.3.0](https://pytorch.org/)
- [DGL==0.5.3](https://www.dgl.ai/pages/start.html)
- sklearn

# Dataset
the ACM raw data, the preprocessing process you can refer to https://github.com/dmlc/dgl/blob/master/examples/pytorch/han/utils.py and https://github.com/seongjunyun/Graph_Transformer_Networks/blob/master/Data_Preprocessing.ipynb (we exacted it to ours ACM_preprocessing.py)

# Run
python main.py
