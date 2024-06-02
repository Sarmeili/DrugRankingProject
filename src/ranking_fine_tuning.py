from datahandler.ctrp_handler import CTRPHandler
from modelexperiment.graphtransformers import DrugRank
import torch
import warnings
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from modelutils.loss_functions import ListAllLoss, LambdaLossLTR

warnings.filterwarnings('ignore')
with open('../configs/config.json') as config_file:
    config = json.load(config_file)
device = config['main']['device']

dh = CTRPHandler()

x_cmpd = dh.get_cmpd_x()
x_cll = dh.get_cll_x()
y = dh.get_reg_y()

total_len = len(x_cmpd)
list_size = 5
initial_train_len = int(len(x_cmpd) * 0.9)
train_len = (initial_train_len // (5 * list_size)) * (5 * list_size)
if train_len > total_len:
    train_len = (total_len // (5 * list_size)) * (5 * list_size)


x_cmpd_train = x_cmpd[:train_len]
x_cmpd_test = x_cmpd[train_len:]

x_cll_train = x_cll[:train_len]
x_cll_test = x_cll[train_len:]

y_train = y[:train_len]
y_test = y[train_len:]


model = DrugRank(3451, 27, 38)
model = model.to(device)
model.load_state_dict(torch.load('../models/official_second.pth'))
loss_fn = LambdaLossLTR()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
list_size = 5
epochs = 20
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

    loader_cmpd_train = dh.load_cmpd(np.take(x_cmpd_train, train_indices, axis=0))
    loader_cll_train = dh.load_cll(np.take(x_cll_train, train_indices, axis=0))
    loader_y_train = dh.load_y(np.take(y_train, train_indices, axis=0))

    loader_cmpd_val = dh.load_cmpd(np.take(x_cmpd_train, val_indices, axis=0))
    loader_cll_val = dh.load_cll(np.take(x_cll_train, val_indices, axis=0))
    loader_y_val = dh.load_y(np.take(y_train, val_indices, axis=0))
    for i in tqdm(range(epochs)):
        model.train()
        for batch_cll, batch_cmpd, batch_y in zip(loader_cll_train, loader_cmpd_train, loader_y_train):
            y_pred = model(batch_cll.to(torch.float32).to(device), batch_cmpd.to(device))
            y_pred = y_pred.reshape(-1, list_size)
            batch_y = batch_y.reshape(-1, list_size)
            loss = loss_fn(batch_y.to(torch.float32).to(device), y_pred.to(torch.float32))
            optimizer.zero_grad()
            loss.requires_grad = True
            loss.backward()
            optimizer.step()

        hist_train.append(loss)

        model.eval()
        with torch.no_grad():
            for batch_cll, batch_cmpd, batch_y in zip(loader_cll_val, loader_cmpd_val, loader_y_val):
                y_pred = model(batch_cll.to(torch.float32).to(device), batch_cmpd.to(device))
                y_pred = y_pred.reshape(-1, list_size)
                batch_y = batch_y.reshape(-1, list_size)
                loss = loss_fn(batch_y.to(torch.float32).to(device), y_pred.to(torch.float32))
            hist_val.append(loss)

torch.save(model.state_dict(), '../models/trained_reg_model.pth')
hist_train = [loss.item() for loss in hist_train]
hist_val = [loss.item() for loss in hist_val]
print(hist_train)
print(hist_val)
plt.figure(1)
plt.plot(hist_train, label='Training Loss')
plt.plot(hist_val, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss on train and validation')
plt.legend()
plt.savefig('../imgs/loss_rank.png')
plt.close()