from datahandler.ctrp_handler import CTRPHandler
from modelexperiment.ae import CllGraphAutoencoder
from modelutils.loss_functions import ListAllLoss
import torch
import warnings
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.data import DataLoader

warnings.filterwarnings('ignore')
with open('../configs/config.json') as config_file:
    config = json.load(config_file)
device = config['main']['device']

dh = CTRPHandler()

x_cll = dh.get_cll_graph_x()
x_cll_train = x_cll[:int(len(x_cll) * 0.9)]
loader_cll_train = dh.load_cll(x_cll_train)

x_cll_val = x_cll[int(len(x_cll) * 0.9):]
loader_cll_val = dh.load_cll(x_cll_val)

model = CllGraphAutoencoder(1,1000)
model = model.to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
epochs = 10
hist_train = []
hist_val = []
for i in tqdm(range(epochs)):
    model.train()
    for batch_cll in loader_cll_train:
        # print(batch_cmpd)
        y_pred, _ = model(batch_cll.to(device))
        loss = loss_fn(y_pred.to(torch.float32), batch_cll.to(device).x.to(torch.float32))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    hist_train.append(loss)

    model.eval()
    with torch.no_grad():
        for batch_cll in loader_cll_val:
            y_pred, _ = model(batch_cll.to(device))
            loss = loss_fn(y_pred.to(torch.float32), batch_cll.to(device).x)

        hist_val.append(loss)

torch.save(model.state_dict(), '../models/graphcll_trained_ae.pth')
hist_train = [loss.item() for loss in hist_train]
hist_val = [loss.item() for loss in hist_val]
plt.figure(1)
plt.plot(hist_train, label='Training Loss')
plt.plot(hist_val, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss on train and validation')
plt.legend()
plt.savefig('../imgs/loss_ae_graphcll.png')
plt.close()