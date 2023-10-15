from modelexperiment.simple_mlp import Model1
from datahandler.ctrp_drugranker import CTRPHandler
from modelutils.training import TrainModel
from modelutils.evaluation import EvaluateModel
import torch

data = CTRPHandler('DrugRanker', 0.2)
train_cll, test_cll = data.create_train_test_cll()
train_drug, test_drug = data.create_train_test_drug()
train_label, test_label = data.create_train_test_label()

model = Model1(train_cll.shape[1], train_drug.shape[1])
model = model.to('cuda')
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 100

trainer = TrainModel(model, train_cll, train_drug, train_label, loss_fn, optimizer, num_epochs)
trained_model = trainer.train_model()

evaluator = EvaluateModel(trained_model, test_cll, test_drug, test_label, loss_fn)
final_loss = evaluator.evaluate()
print('Final Loss : '+ str(final_loss))