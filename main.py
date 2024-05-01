import torch
from datahandler.ctrp_handler import CTRPHandler
import warnings
from torch.utils.data import DataLoader
import numpy as np
import json
from rdkit import Chem
import torch_geometric as tg
from modelexperiment.official_first import DrugRank
import pandas as pd
from modelutils.loss_functions import MyListNetLoss, LambdaLossLTR
from tqdm import tqdm
import matplotlib.pyplot as plt
from datawrangling.wrangling import Wrangler
from sklearn.metrics import ndcg_score

warnings.filterwarnings('ignore')
with open('config.json') as config_file:
    config = json.load(config_file)
wrangle_data = config['data_wrangling']['wrangle_data']
netprop = config['network_propagation']['is_netprop']

if wrangle_data:
    wrangle = Wrangler()
    wrangle.save_wrangled_data()
if netprop:
    netprop = CTRPHandler()
    netprop.netprop_dim_reduction()

model = DrugRank(1, 27, 1)
model = model.to('cuda')

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = LambdaLossLTR()
epochs = 30
list_size = 5
history_train = []
history_test = []
ndcg_train = []
ndcg_test = []
data = CTRPHandler()
exp_cll_df, edges, edges_attrib = data.select_gene_feature()
rank_train, rank_val, rank_test = data.listwise_ranking_df()
for i in tqdm(range(epochs)):
    model.train()
    for item in data.generate_cll_drug_response(rank_train):
        clls, drugs_list, responses_list = item
        for cll, drugs, responses in zip(clls, drugs_list, responses_list):
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
            responses = torch.reshape(responses, (1, -1))
            loss = loss_fn(y_pred, responses)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            del mol_graph
            del cll_graph
            del bio_graph
    history_train.append(loss)
    ndcg_train.append(ndcg_score(responses.cpu().detach().numpy(), y_pred.cpu().detach().numpy()))
    model.eval()
    with torch.no_grad():
        for item in data.generate_cll_drug_response(rank_val):
            clls, drugs_list, responses_list = item

            for cll, drugs, responses in zip(clls, drugs_list, responses_list):
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
                responses = torch.reshape(responses, (1, -1))
                loss = loss_fn(y_pred, responses)
                del mol_graph
                del cll_graph
                del bio_graph
    history_test.append(loss)
    ndcg_test.append(ndcg_score(responses.cpu().detach().numpy(), y_pred.cpu().detach().numpy()))

torch.save(model, 'models/official_first.pt')
history_train = [loss.item() for loss in history_train]
history_test = [loss.item() for loss in history_test]
plt.figure(1)
plt.plot(history_train, label='Training Loss')
plt.plot(history_test, label='Testing Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss on train and test')
plt.legend()
plt.savefig('loss_plot.png')
plt.show()

plt.figure(2)
plt.plot(ndcg_train, label='Training NDCG')
plt.plot(ndcg_test, label='Testing NDCG')
plt.xlabel('Epoch')
plt.ylabel('NDCG')
plt.title('NDCG on train and test')
plt.legend()
plt.savefig('metric_plot.png')
plt.show()





