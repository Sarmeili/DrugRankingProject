import torch
from datahandler.ctrp_drugranker import CTRPHandler
print(torch.cuda.is_available())
print(torch.__version__)

data = CTRPHandler('DrugRanker', 0.1)
cll_feat = data.create_tensor_feat_cll()
drug_feat = data.create_tensor_feat_drug()
print(cll_feat.shape)
print(drug_feat.shape)
'''pca = torch.pca_lowrank(cll_feat, q=1000)
print(pca[0])
print(pca[1])
print(pca[2])
print(pca[0].shape)'''