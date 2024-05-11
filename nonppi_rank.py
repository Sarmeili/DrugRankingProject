from datahandler.ctrp_handler import CTRPHandler
from modelexperiment.official_second import DrugRank
import torch
import warnings
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pickle
from torch_geometric.data import DataLoader
from modelutils.loss_functions import ListAllLoss, LambdaLossLTR

warnings.filterwarnings('ignore')
with open('config.json') as config_file:
    config = json.load(config_file)
device = config['main']['device']
batch_size = config['datahandler']['ctrp_handler']['batch_size']

def load_data(input, batch_size):
    n_samples = len(input)
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        yield input[start:end]


dh = CTRPHandler()

cmpds = dh.get_cmpdran_x()
with open('drugs.pk1', 'wb') as f:
    pickle.dump(cmpds, f)

with open('drugs.pk1', 'rb') as f:
    cmpds = pickle.load(f)


x_cmpd = cmpds
x_cll = dh.get_cllran_x()
y = dh.get_rank_y()

x_cmpd_train = x_cmpd[:int(len(x_cmpd) * 0.9)]
x_cll_train = x_cll[:int(len(x_cll) * 0.9)]
y_train = y[:int(len(y) * 0.9)]

x_cmpd_test = x_cmpd[int(len(x_cmpd) * 0.9):]
x_cll_test = x_cll[int(len(x_cll) * 0.9):]
y_test = y[int(len(y) * 0.9):]


model = DrugRank(3451, 27)
model = model.to(device)
loss_fn = LambdaLossLTR()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
epochs = 30
hist_train = []
hist_val = []
k = 5
fold_size = len(y_train)//k

for fold in range(k):
    start_val = fold * fold_size
    end_val = start_val + fold_size
    indices = list(range(len(y_train)))
    val_indices = indices[start_val:end_val]
    train_indices = indices[:start_val] + indices[end_val:]

    x_cmpd_trn = [x_cmpd_train[i] for i in train_indices]
    loader_cmpd_train = load_data(x_cmpd_trn, batch_size=batch_size)
    loader_cll_train = dh.load_cll(np.take(x_cll_train, train_indices, axis=0))
    y_trn = [y_train[i] for i in train_indices]
    loader_y_train = load_data(y_trn, batch_size=batch_size)

    x_cmpd_val = [x_cmpd_train[i] for i in val_indices]
    loader_cmpd_val = load_data(x_cmpd_val, batch_size=batch_size)
    loader_cll_val = dh.load_cll(np.take(x_cll_train, val_indices, axis=0))
    y_val = [y_train[i] for i in val_indices]
    loader_y_val = load_data(y_val, batch_size=batch_size)

    for i in tqdm(range(epochs)):
        model.train()
        for batch_cll, batch_cmpd, batch_y in zip(loader_cll_train, loader_cmpd_train, loader_y_train):
            y_batch_pred = []
            for cll, cmpds, ys in zip(batch_cll, batch_cmpd, batch_y):
                y_preds = []
                for cmpd, y in zip(cmpds, ys):
                    y_pred = model(cll.to(torch.float32).to(device).reshape(1, -1), cmpd.to(device))
                    y_preds.append(y_pred)
                y_batch_pred.append(y_preds)

            y_batch_pred = torch.tensor(y_batch_pred)
            batch_y = torch.tensor(batch_y)
            loss = loss_fn(batch_y.to(torch.float32).to(device), y_batch_pred.to(torch.float32).to(device))
            optimizer.zero_grad()
            loss.requires_grad = True
            loss.backward()
            optimizer.step()

        hist_train.append(loss)

        model.eval()
        with torch.no_grad():
            for batch_cll, batch_cmpd, batch_y in zip(loader_cll_val, loader_cmpd_val, loader_y_val):
                y_batch_pred = []
                for cll, cmpds, ys in zip(batch_cll, batch_cmpd, batch_y):
                    y_preds = []
                    for cmpd, y in zip(cmpds, ys):
                        y_pred = model(cll.to(torch.float32).to(device).reshape(1, -1), cmpd.to(device))
                        y_preds.append(y_pred)
                    y_batch_pred.append(y_preds)
                y_batch_pred = torch.tensor(y_batch_pred)
                batch_y = torch.tensor(batch_y)
                loss = loss_fn(batch_y.to(torch.float32).to(device), y_batch_pred.to(torch.float32).to(device))
            hist_val.append(loss)

torch.save(model.state_dict(), 'models/official_second_rank.pth')
hist_train = [loss.item() for loss in hist_train]
hist_val = [loss.item() for loss in hist_val]
plt.figure(1)
plt.plot(hist_train, label='Training Loss')
plt.plot(hist_val, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss on train and validation')
plt.legend()
plt.savefig('cll_without_graph_rank.png')
plt.close()