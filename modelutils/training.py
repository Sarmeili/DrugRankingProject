import torch
from tqdm import tqdm
import json
import torch_geometric as tg


class TrainModel:
    def __init__(self, model, train_feat):
        with open('config.json') as config_file:
            config = json.load(config_file)
        self.model = model
        self.train_feat = train_feat
        loss_type = config['training_hp']['loss_fn']
        opt_type = config['training_hp']['optimizer']['optim_kind']
        self.lr = config['training_hp']['optimizer']['lr']
        if loss_type == 'MSE':
            self.loss_fn = torch.nn.MSELoss()
        if opt_type == 'adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.num_epochs = config['training_hp']['num_epochs']
        self.drug_feat = config['datahandler']['ctrp_drugranker']['drug_feat']

    def train_model(self):
        for n in tqdm(range(self.num_epochs)):
            for train_X, train_label in self.train_feat:
                train_exp = train_X[0]
                train_mut = train_X[1]
                train_drug = train_X[2]
                if 'graph' in self.drug_feat:
                    loader = tg.loader.DataLoader(train_drug, batch_size=len(train_drug))
                    train_drug = next(iter(loader))
                    train_drug.x = torch.tensor(train_drug.x, dtype=torch.float32)
                    train_drug = train_drug.to('cuda')
                    train_exp = train_exp.to('cuda')
                    train_label = train_label.to('cuda')
                y_pred = self.model(train_exp, train_drug)
                loss = self.loss_fn(y_pred, train_label)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()
        return self.model
