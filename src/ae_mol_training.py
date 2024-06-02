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
with open('../configs/config.json') as config_file:
    config = json.load(config_file)
device = config['main']['device']

dh = CTRPHandler()

x_cmpd = dh.get_cmpd_x()
x_cmpd_train = x_cmpd[:int(len(x_cmpd) * 0.9)]
loader_cmpd_train = dh.load_cmpd(x_cmpd_train)

x_cmpd_val = x_cmpd[int(len(x_cmpd) * 0.9):]
loader_cmpd_val = dh.load_cmpd(x_cmpd_val)

model = MolGraphAutoencoder(27,100)
model = model.to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.1)
epochs = 30
hist_train = []
hist_val = []
for i in tqdm(range(epochs)):
    model.train()
    for batch_cmpd in loader_cmpd_train:
        y_pred, _ = model(batch_cmpd.to(device))
        loss = loss_fn(y_pred.to(torch.float32), batch_cmpd.to(device).x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    hist_train.append(loss)

    model.eval()
    with torch.no_grad():
        for batch_cmpd in loader_cmpd_val:
            y_pred, _ = model(batch_cmpd.to(device))
            loss = loss_fn(y_pred.to(torch.float32), batch_cmpd.to(device).x)

        hist_val.append(loss)

torch.save(model.state_dict(), 'models/mol_trained_ae.pth')
hist_train = [loss.item() for loss in hist_train]
hist_val = [loss.item() for loss in hist_val]
plt.figure(1)
plt.plot(hist_train, label='Training Loss')
plt.plot(hist_val, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss on train and validation')
plt.legend()
plt.savefig('loss_ae_mol.png')
plt.close()