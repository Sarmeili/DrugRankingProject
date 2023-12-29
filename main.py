import torch
from datahandler.ctrp_handler import CTRPHandler
from datahandler.ctrp_handler import CTRPDatasetTorch
import warnings
from torch.utils.data import DataLoader
import numpy as np
import json
from rdkit import Chem
import torch_geometric as tg
from torch_geometric.utils import scatter
from modelexperiment.official_first import DrugRank
from torch_geometric.utils import scatter
from modelutils.training_reg import TrainModel
import pandas as pd
from torch_geometric.utils import degree
from datahandler.netprop import NetProp
from modelutils.loss_functions import LambdaMARTLoss, LambdaLoss
from tqdm import tqdm
import matplotlib.pyplot as plt
from datawrangling.wrangling import Wrangler

warnings.filterwarnings('ignore')
with open('config.json') as config_file:
    config = json.load(config_file)
wrangle_data = config['data_wrangling']['wrangle_data']
netprop = config['network_propagation']['is_netprop']

if wrangle_data:
    wrangle = Wrangler()
    wrangle.save_wrangled_data()
if netprop:
    netprop = CTRPHandler([0, 1])
    netprop.netprop_dim_reduction()

model = DrugRank(1, 27, 1)
model = model.to('cuda')

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = LambdaMARTLoss()
epochs = 50
history_train = []
history_test = []
data = CTRPHandler(data_volume=[0, 1])
exp_cll_df, edges, edges_attrib = data.select_gene_feature()
rank_train, rank_val, rank_test = data.listwise_ranking_df()
for i in tqdm(range(epochs)):
    for item in data.generate_cll_drug_response(rank_train):
        clls, drugs_list, responses_list = item
        for cll, drugs, responses in zip(clls, drugs_list, responses_list):
            list_size = 5
            indices = torch.randperm(len(drugs))[:list_size]
            drugs = torch.tensor(drugs)[indices].tolist()
            responses = torch.tensor(responses)[indices].to('cuda')
            y_pred = torch.tensor([]).to('cuda')
            for drug in drugs:
                mol_graph = data.create_mol_graph(drug)
                mol_graph = mol_graph.to('cuda')
                cll_graph, bio_graph = data.create_cll_bio_graph(drug, cll, exp_cll_df, edges, edges_attrib)
                cll_graph = cll_graph.to('cuda')
                bio_graph = bio_graph.to('cuda')
                y_pred_drug = model(cll_graph, mol_graph, bio_graph)
                y_pred = torch.concat((y_pred_drug, y_pred), dim=1)
            responses = torch.reshape(responses, (1, list_size))
            loss = loss_fn(y_pred, responses)
            history_train.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    for item in data.generate_cll_drug_response(rank_val):
        clls, drugs_list, responses_list = item
        model.eval()
        for cll, drugs, responses in zip(clls, drugs_list, responses_list):
            list_size = 5
            indices = torch.randperm(len(drugs))[:list_size]
            drugs = torch.tensor(drugs)[indices].tolist()
            responses = torch.tensor(responses)[indices].to('cuda')
            y_pred = torch.tensor([]).to('cuda')
            for drug in drugs:
                mol_graph = data.create_mol_graph(drug)
                mol_graph = mol_graph.to('cuda')
                cll_graph, bio_graph = data.create_cll_bio_graph(drug, cll, exp_cll_df, edges, edges_attrib)
                cll_graph = cll_graph.to('cuda')
                bio_graph = bio_graph.to('cuda')
                y_pred_drug = model(cll_graph, mol_graph, bio_graph)
                y_pred = torch.concat((y_pred_drug, y_pred), dim=1)
            responses = torch.reshape(responses, (1, list_size))
            loss = loss_fn(y_pred, responses)
            history_test.append(loss)

torch.save(model, 'models/official_first.pt')
history_train = [loss.item() for loss in history_train]
history_test = [loss.item() for loss in history_test]
plt.plot(range(1, len(history_train) + 1), history_train, label='Training Loss')
plt.plot(range(1, len(history_test) + 1), history_test, label='Testing Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.savefig('loss_plot.png')
plt.show()





