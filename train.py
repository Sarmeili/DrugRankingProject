from modelexperiment.simple_mlp import Model1
from datahandler.ctrp_drugranker import CTRPHandler
from datahandler.ctrp_drugranker import CTRPDatasetTorch
from modelutils.training import TrainModel
from modelutils.evaluation import EvaluateModel
from torch.utils.data import DataLoader
import torch
import warnings

warnings.filterwarnings('ignore')

data = CTRPHandler('DrugRanker', 1)
train_dataset = CTRPDatasetTorch(data, True)
train_dataloader = DataLoader(train_dataset, batch_size=64)
test_dataset = CTRPDatasetTorch(data, False)
test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset))

model = Model1(train_dataset[0][0][0].shape[0], train_dataset[0][0][1].shape[0])
model = model.to('cuda')
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 100
trainer = TrainModel(model, train_dataloader, loss_fn, optimizer, num_epochs)
trained_model = trainer.train_model()
torch.save(trained_model, 'models/simple_mpl.pt')
# trained_model = torch.load('models/simple_mpl.pt')
for test_X, test_label in test_dataloader:
    test_cll, test_drug = test_X
    y_pred = trained_model(test_cll, test_drug)
    trained_model.eval()
    print('y_pred', y_pred)
    print('y_label', test_label)
    evaluator = EvaluateModel(trained_model, test_cll, test_drug, test_label, loss_fn)
    final_loss = evaluator.evaluate()
    print('Final Loss : ' + str(final_loss))