from datahandler.ctrp_handler import CTRPHandler
# from modelexperiment.graphtransformers import DrugRank
# from modelexperiment.gat_drp import DrugRank
from modelexperiment.customgcn_drp import DrugRank
# from modelexperiment.nnconv_drp import DrugRank
import torch
import warnings
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from torch_geometric.data import DataLoader


warnings.filterwarnings('ignore')
with open('../configs/config.json') as config_file:
    config = json.load(config_file)
device = config['main']['device']
batch_size = config['datahandler']['ctrp_handler']['batch_size']

dh = CTRPHandler()

x_cmpd = dh.get_cmpd_x()
x_cll = dh.get_cll_x()
y = dh.get_reg_y()
weight = dh.get_reg_weigth()

x_cmpd_train = x_cmpd[:int(len(x_cmpd) * 0.9)]
x_cll_train = x_cll[:int(len(x_cll) * 0.9)]
y_train = y[:int(len(y) * 0.9)]
weight_train = weight[:int(len(weight) * 0.9)]

x_cmpd_test = x_cmpd[int(len(x_cmpd) * 0.9):]
x_cll_test = x_cll[int(len(x_cll) * 0.9):]
y_test = y[int(len(y) * 0.9):]
weight_test = weight[int(len(weight) * 0.9):]

def weighted_loss(output, target, weights):
    loss = loss_fn(output, target)
    weighted_loss = loss * weights  # Element-wise multiplication
    return weighted_loss.mean()


model = DrugRank(3451, 27, 38) # , 38
model = model.to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=0.001)
epochs = 3
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
    loader_weight_train = dh.load_weight(np.take(weight_train, train_indices, axis=0))

    loader_cmpd_val = dh.load_cmpd(np.take(x_cmpd_train, val_indices, axis=0))
    loader_cll_val = dh.load_cll(np.take(x_cll_train, val_indices, axis=0))
    loader_y_val = dh.load_y(np.take(y_train, val_indices, axis=0))
    loader_weight_val = dh.load_weight(np.take(weight_train, val_indices, axis=0))
    for i in tqdm(range(epochs)):
        model.train()
        for batch_cll, batch_cmpd, batch_weight, batch_y in zip(loader_cll_train, loader_cmpd_train,
                                                                loader_weight_train, loader_y_train):

            y_pred = model(batch_cll.to(torch.float32).to(device), batch_cmpd.to(device))
            batch_y = batch_y.reshape(-1, 1)
            loss = weighted_loss(batch_y.to(torch.float32).to(device), y_pred.to(torch.float32),
                                 batch_weight.to(torch.float32).to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        hist_train.append(loss)

        model.eval()
        with torch.no_grad():
            for batch_cll, batch_cmpd, batch_weight, batch_y in zip(loader_cll_val, loader_cmpd_val,
                                                                    loader_weight_val, loader_y_val):
                y_pred = model(batch_cll.to(torch.float32).to(device), batch_cmpd.to(device))
                batch_y = batch_y.reshape(-1, 1)
                loss = weighted_loss(batch_y.to(torch.float32).to(device), y_pred.to(torch.float32),
                                     batch_weight.to(torch.float32).to(device))
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
plt.savefig('../imgs/loss_reg.png')
plt.close()

cll_test = DataLoader(x_cll_test, batch_size=batch_size)
cmpd_test = DataLoader(x_cmpd_test, batch_size=batch_size)
y_test_pred_list = torch.tensor([]).to('cpu')
model.eval()
with torch.no_grad():
    for cll, cmpd in zip(cll_test, cmpd_test):
        y_test_pred = model(cll.to(torch.float32).to(device), cmpd.to(device))
        y_test_pred = y_test_pred.to('cpu')
        y_test_pred_list = torch.cat((y_test_pred, y_test_pred_list), 0).to('cpu')
plt.figure(figsize=(8, 6))
y_test_pred_list = y_test_pred_list.detach().cpu().numpy()
plt.plot(y_test, y_test, label='True values (ideal line)', color='blue', linestyle='--')
plt.scatter(y_test, y_test_pred_list, label='Predicted', color='red')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values')
plt.legend()
plt.grid(True)
plt.savefig('../imgs/truevspred.png')