import torch
from datahandler.ctrp_drugranker import CTRPHandler
from datahandler.ctrp_drugranker import CTRPDatasetTorch
import warnings
#from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
import numpy as np
import json
from rdkit import Chem
import torch_geometric as tg
warnings.filterwarnings('ignore')

data = CTRPHandler(data_volume=[0, 0.01])
drug_feat = data.create_tensor_feat_drug()
print(len(drug_feat))
print(torch.cuda.is_available())
print(torch.__version__)
'''tg_list = []
for smile in cmpd_df['cpd_smiles']:
    tg_list.append(tg.utils.from_smiles(smile))
loader = DataLoader(tg_list, batch_size=8)
for X in loader:
    print(X)
    break
'''