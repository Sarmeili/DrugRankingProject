import torch
import json
import torch_geometric as tg


class EvaluateModel:
    def __init__(self, model, test_exp, test_mut, test_drug, test_label):
        with open('config.json') as config_file:
            config = json.load(config_file)
        self.model = model
        self.test_exp = test_exp
        self.test_mut = test_mut
        self.test_drug = test_drug
        self.test_label = test_label
        loss_type = config['training_hp']['loss_fn']
        if loss_type == 'MSE':
            self.loss_fn = torch.nn.MSELoss()
        self.drug_feat = config['datahandler']['ctrp_drugranker']['drug_feat']

    def evaluate(self):
        self.model.eval()
        if 'graph' in self.drug_feat:
            loader = tg.loader.DataLoader(self.test_drug, batch_size=len(self.test_drug))
            self.test_drug = next(iter(loader))
        y_pred_test = self.model(self.test_exp, self.test_drug)
        test_loss = self.loss_fn(y_pred_test, self.test_label)
        return test_loss
