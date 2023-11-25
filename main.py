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
'''q = data.bio_dimreduction(rank)
netprop = NetProp()
ind = netprop.netpropagete(q[0])
print(ind)'''
mol, bio = data.listwise_drug_molecule_bio(rank)
print(mol)
print(bio[0][0])
print(bio[0][0].x)
#print(bio[0][0].edge_index)
#print(bio[0][0].edge_attr)
print(bio[0][0].is_undirected())
print(bio[0][0].has_isolated_nodes())

