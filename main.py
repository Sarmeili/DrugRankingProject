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
import pandas as pd
from torch_geometric.utils import degree
from datahandler.netprop import NetProp

warnings.filterwarnings('ignore')
print(torch.cuda.is_available())
data = CTRPHandler(data_volume=[0, 0.1])
rank = data.listwise_ranking_df()
'''mol, bio = data.listwise_drug_molecule_bio(rank)
print(mol)
print(bio)'''
drug_chosen = data.dim_reduction()
print(drug_chosen)




