import torch
import torch_geometric as tg
import json


class GCNMol(torch.nn.Module):
    def __init__(self, cll_input_layer_size, drug_input_layer_size):
        super(GCNMol, self).__init__()
        with open('config.json') as config_file:
            config = json.load(config_file)
        last_layer = config['model_experiments']['graphmol_mlp']['last_layer']
        self.cll_input_layer_size = cll_input_layer_size

        self.conv1_drug = tg.nn.GCNConv(drug_input_layer_size, 200)
        self.conv2_drug = tg.nn.GCNConv(200, 200)
        self.linear_drug = tg.nn.Linear(200, last_layer)

    def forward(self, train_cll, train_drug):


        return concat_data
