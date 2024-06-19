import pandas as pd
import numpy as np
from datahandler.cll_graph_handler import CllGraphHandler
import pickle
from torch_geometric.data import DataLoader
from modelexperiment.biggraphcllgcn import GCNAutoencoder
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


dh = CllGraphHandler()
cll_graph, cllname_list = dh.get_graph()
with open('graph_list.pkl', 'wb') as f:
    pickle.dump(cll_graph, f)

device = 'cuda'
train_dataset, val_dataset = train_test_split(cll_graph, test_size=0.2, random_state=42)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

input_dim = 12
embedding_dim = 128
model = GCNAutoencoder(input_dim=input_dim, embedding_dim=embedding_dim).to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 30
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        output, h = model(data.to(device))
        loss = criterion(output, data.x)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_losses.append(train_loss / len(train_loader))

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            output, h = model(data)
            loss = criterion(output, data.x)
            val_loss += loss.item()
    val_losses.append(val_loss / len(val_loader))

    print(f'Epoch {epoch+1}, Train Loss: {train_losses[-1]}, Validation Loss: {val_losses[-1]}')

torch.save(model.state_dict(), '../models/biggraphcll_ae.pth')
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')
plt.savefig('../imgs/loss_ae_biggraphcll.png')
plt.close()

feature_list = []
for graph in cll_graph:
    _, h = model(graph)
    feature_list.append(h.detach().numpy())

pd.DataFrame(np.squeeze(feature_list), index=cllname_list).to_csv('../data/cllfeats.csv')