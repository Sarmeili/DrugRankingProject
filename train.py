from modelexperiment.simple_mlp import Model1
from datahandler.ctrp_drugranker import CTRPHandler
import torch

data = CTRPHandler('DrugRanker', 0.05)
cll_data = data.create_tensor_feat_cll()
drug_data = data.create_tensor_feat_drug()
cll_data = cll_data.to('cuda').to(torch.float32)
drug_data = drug_data.to('cuda').to(torch.float32)
model = Model1(cll_data.shape[1] ,drug_data.shape[1])
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
num_epoch = 100