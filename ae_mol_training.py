from datahandler.ctrp_handler import CTRPHandler
from modelexperiment.ae import MolGraphAutoencoder
from modelutils.loss_functions import ListAllLoss
import torch
import warnings
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.data import DataLoader

warnings.filterwarnings('ignore')
with open('config.json') as config_file:
    config = json.load(config_file)
device = config['main']['device']

dh = CTRPHandler()

x_cmpd = dh.get_cmpd_x()
x_cmpd = x_cmpd[:int(len(x_cmpd) * 0.9)]
loader_cmpd_train = dh.load_cmpd(x_cmpd)

x_cll = dh.get_cll_x()
x_cll = x_cll[:int(len(x_cll) * 0.9)]
loader_cll_train = dh.load_cll(x_cll)


y = dh.get_reg_y()
y = y[:int(len(y) * 0.9)]
loader_y_train = dh.load_y(y)

weight = dh.get_reg_weigth()
weight = weight[:int(len(weight) * 0.9)]
loader_weight_train = dh.load_weight(weight)


x_cmpd = dh.get_cmpd_x()
x_cmpd = x_cmpd[int(len(x_cmpd) * 0.9):]
loader_cmpd_test = dh.load_cmpd(x_cmpd)

x_cll = dh.get_cll_x()
x_cll = x_cll[int(len(x_cll) * 0.9):]
loader_cll_test = dh.load_cll(x_cll)

y = dh.get_reg_y()
y = y[int(len(y) * 0.9):]
loader_y_test = dh.load_y(y)

weight = dh.get_reg_weigth()
y = y[int(len(y) * 0.9):]
loader_weight_test = dh.load_y(y)

model = MolGraphAutoencoder(27,100)
model = model.to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
epochs = 10
hist_train = []
hist_val = []
for i in tqdm(range(epochs)):
    model.train()
    for batch_cll, batch_cmpd, batch_weight, batch_y in zip(loader_cll_train, loader_cmpd_train,
                                                            loader_weight_train, loader_y_train):
        # print(batch_cmpd)
        y_pred, _ = model(batch_cmpd.to(device))
        loss = loss_fn(y_pred.to(torch.float32), batch_cmpd.to(device).x)
        '''loss = weighted_loss(batch_y.to(torch.float32).to(device), y_pred.to(torch.float32),
                             batch_weight.to(torch.float32).to(device))'''
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    hist_train.append(loss)

    model.eval()
    with torch.no_grad():
        for batch_cll, batch_cmpd, batch_weight, batch_y in zip(loader_cll_test, loader_cmpd_test,
                                                                loader_weight_test, loader_y_test):
            y_pred, _ = model(batch_cmpd.to(device))
            loss = loss_fn(y_pred.to(torch.float32), batch_cmpd.to(device).x)

        hist_val.append(loss)

hist_train = [loss.item() for loss in hist_train]
hist_val = [loss.item() for loss in hist_val]
plt.figure(1)
plt.plot(hist_train, label='Training Loss')
plt.plot(hist_val, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss on train and validation')
plt.legend()
plt.savefig('cll_without_graph.png')
plt.close()