import torch
import torch_geometric as tg
import json


class DrugRank(torch.nn.Module):
    def __init__(self, cll_size, mol_size, bio_siz):
        super(DrugRank, self).__init__()
        with open('config.json') as config_file:
            config = json.load(config_file)
        self.mol_ll = config['model_experiments']['drugrank']['mol_ll']
        self.bio_ll = config['model_experiments']['drugrank']['bio_ll']
        self.cll_ll = config['model_experiments']['drugrank']['cll_ll']

        self.conv1_mol = tg.nn.GCNConv(mol_size, 200)
        self.conv2_drug = tg.nn.GCNConv(200, 200)
        self.linear_drug = tg.nn.Linear(200, self.mol_ll)

    def forward(self, train_cll, train_drug):


        return concat_data
