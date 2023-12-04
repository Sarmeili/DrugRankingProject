from modelexperiment.graphmol_mlp import GCNMol
from datahandler.ctrp_drugranker import CTRPHandler
from datahandler.ctrp_drugranker import CTRPDatasetTorch
from modelutils.training_reg import TrainModel
from modelutils.evaluation import EvaluateModel
from torch.utils.data import DataLoader
import torch_geometric as tg
import torch
import numpy as np
import warnings


warnings.filterwarnings('ignore')

data = CTRPHandler(data_volume=[0, 0.01])
exp_shape = data.get_exp_dim()
drug_shape = data.get_drug_dim()
model = GCNMol(exp_shape, drug_shape)
model = model.to('cuda')

del data


for i in np.arange(0, 1, 0.01):

    print(i)
    data = CTRPHandler(data_volume=[i, i+0.01])
    train_dataset = CTRPDatasetTorch(data, True)
    train_dataloader = tg.loader.DataLoader(train_dataset, batch_size=64)
    test_dataset = CTRPDatasetTorch(data, False)
    test_dataloader = tg.loader.DataLoader(test_dataset, batch_size=len(test_dataset))
    trainer = TrainModel(model, train_dataloader)
    model = trainer.train_model()
    for test_X, test_label in test_dataloader:
        test_exp, test_mut, test_drug = test_X
        y_pred = model(test_exp, test_drug)
        model.eval()
        evaluator = EvaluateModel(model, test_exp, test_mut, test_drug, test_label)
        final_loss = evaluator.evaluate()
        print('Final Loss : ' + str(final_loss))

    del train_dataset
    del train_dataloader
    del test_dataset
    del test_dataloader

torch.save(model, 'models/simple_mpl.pt')

'''
data = CTRPHandler(data_volume=[0, 0.01])
exp_shape = data.get_exp_dim()
drug_shape = data.get_drug_dim()
model = Model1(exp_shape, drug_shape)
model = model.to('cuda')

del data


for i in np.arange(0, 1, 0.01):

    print(i)
    data = CTRPHandler(data_volume=[i, i+0.01])
    train_dataset = CTRPDatasetTorch(data, True)
    train_dataloader = DataLoader(train_dataset, batch_size=64)
    test_dataset = CTRPDatasetTorch(data, False)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset))
    trainer = TrainModel(model, train_dataloader)
    model = trainer.train_model()
    for test_X, test_label in test_dataloader:
        test_exp, test_mut, test_drug = test_X
        y_pred = model(test_exp, test_drug)
        model.eval()
        evaluator = EvaluateModel(model, test_exp, test_mut, test_drug, test_label)
        final_loss = evaluator.evaluate()
        print('Final Loss : ' + str(final_loss))

    del train_dataset
    del train_dataloader
    del test_dataset
    del test_dataloader

torch.save(model, 'models/simple_mpl.pt')
'''
