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
from modelexperiment.official_first import DrugRank
from torch_geometric.utils import scatter
from modelutils.training import TrainModel
import pandas as pd
from torch_geometric.utils import degree
from datahandler.netprop import NetProp

warnings.filterwarnings('ignore')
print(torch.cuda.is_available())
data = CTRPHandler(data_volume=[0, 0.1])
rank = data.listwise_ranking_df()
model = DrugRank(1, 9, 1)
model = model.to('cuda')
print(rank)
for item in data.generate_cll_drug_response(rank):
    clls = item[0]
    drugs_list = item[1]
    responses_list = item[2]
    for cll, drugs, responses in zip(clls, drugs_list, responses_list):
        for drug in drugs:
            mol_graph = data.create_mol_graph(drug)
            mol_graph = mol_graph.to('cuda')
            cll_graph, bio_graph = data.create_cll_bio_graph(drug, cll)
            cll_graph = cll_graph.to('cuda')
            bio_graph = bio_graph.to('cuda')
            print(model(cll_graph, mol_graph, bio_graph))
        print('\n')
    print('\n\n')









