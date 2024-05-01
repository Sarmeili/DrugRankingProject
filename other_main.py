from datahandler.ctrp_handler import CTRPHandler
from modelexperiment.official_second import DrugRank
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
model = DrugRank(3451, 27)
model = model.to(device)
loss_fn = ListAllLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 10
hist = []
for i in tqdm(range(epochs)):
    model.train()
    for batch in dh.get_npgenes_drugs():
        cll, drug, response = batch
        cll = cll.to(torch.float).to(device)
        # drug = drug.to(device)
        drug = DataLoader(drug, batch_size=128)
        response = response.to(device)
        for cmpd in drug:
            y_pred = model(cll, cmpd)
        loss = loss_fn(y_pred, response)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    hist.append(loss)
hist = [loss.item() for loss in hist]
plt.figure(1)
plt.plot(hist, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss on train')
plt.legend()
plt.savefig('new.png')
plt.close()