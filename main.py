import torch
from datahandler.ctrp_drugranker import CTRPHandler
from datahandler.ctrp_drugranker import CTRPDatasetTorch
import warnings
from torch.utils.data import DataLoader
import numpy as np
import json
from rdkit import Chem
import torch_geometric as tg
from torch_geometric.utils import scatter
from modelexperiment.graphmol_mlp import GCNMol
from torch_geometric.utils import scatter
from modelutils.training import TrainModel

warnings.filterwarnings('ignore')

data = CTRPHandler(data_volume=[0, 0.01])
exp_shape = data.get_exp_dim()
drug_shape = data.get_drug_dim()
train_dataset = CTRPDatasetTorch(data, True)
train_dataloader = tg.loader.DataLoader(train_dataset, batch_size=64)
model = GCNMol(exp_shape, drug_shape)
for i in range(2):
    for X, y in train_dataloader:
        train_exp = X[0]
        train_mut = X[1]
        train_drug = X[2]
        loader = tg.loader.DataLoader(train_drug, batch_size=len(train_drug))
        batch = next(iter(loader))
        #batch.x = torch.tensor(batch.x, dtype=torch.float32)
        #batch = batch.to('cpu')
        #train_exp = train_exp.to('cpu')
        y_pred = model(train_exp, batch)
        print(y_pred)
        break

'''drug_feat = data.create_tensor_feat_drug()
perm = torch.randperm(len(drug_feat))
loader = DataLoader(drug_feat, batch_size=16)
model = GCNMol(3000, drug_feat[0].num_features)
for batch in loader:
    print(batch.num_graphs)
    batch.x = torch.tensor(batch.x, dtype=torch.float32)
    out = model(batch)

    print(out.shape)'''

'''print(torch.cuda.is_available())
print(torch.__version__)'''