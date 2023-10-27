import torch
from datahandler.ctrp_drugranker import CTRPHandler
from datahandler.ctrp_drugranker import CTRPDatasetTorch
import warnings
from torch.utils.data import DataLoader
import numpy as np

warnings.filterwarnings('ignore')

print(torch.cuda.is_available())
print(torch.__version__)

data = CTRPHandler('DrugRanker', [0, 0.01])
train_cll, test_cll = data.create_train_test_cll()
train_drug, test_drug = data.create_train_test_drug()
train_label, test_label = data.create_train_test_label()
print(torch.sum(torch.isnan(train_cll)))
print(torch.sum(torch.isnan(test_cll)))
print(torch.sum(torch.isnan(train_drug)))
print(torch.sum(torch.isnan(test_drug)))
print(torch.sum(torch.isnan(train_label)))
print(torch.sum(torch.isnan(test_label)))

'''train_dataset = CTRPDatasetTorch(data, True)
print(train_dataset[0][0][0].shape[0])
print(train_dataset[0][0][1].shape[0])
print(train_dataset[0][1].shape)
train_dataloader = DataLoader(train_dataset, batch_size=64)
# print(next(iter(train_dataloader)))
for X, y in train_dataloader:
    print(X[0].dtype)
    print(X[1].dtype)
    print(torch.concat((X[0], X[1]), 1).dtype)
    print(y.shape)
    break'''


'''pca = torch.pca_lowrank(cll_feat, q=1000)
print(pca[0])
print(pca[1])
print(pca[2])
print(pca[0].shape)
model = torch.load('models/simple_mpl.pt')
print(model)
for parameter in model.parameters():
    print(parameter)'''


