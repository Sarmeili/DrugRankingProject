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

        self.conv1_drug = tg.nn.GCNConv(drug_input_layer_size, 20)
        self.conv2_drug = tg.nn.GCNConv(20, 25)
        self.linear_drug = tg.nn.Linear(25, last_layer)

        self.linear1_ccl = torch.nn.Linear(self.cll_input_layer_size, int(self.cll_input_layer_size * 0.5))
        self.activation1_ccl = torch.nn.ReLU()

        self.linear2_ccl = torch.nn.Linear(int(self.cll_input_layer_size * 0.5), int(self.cll_input_layer_size * 0.25))
        self.activation2_ccl = torch.nn.ReLU()

        self.linear3_ccl = torch.nn.Linear(int(self.cll_input_layer_size * 0.25),
                                           int(self.cll_input_layer_size * 0.125))
        self.activation3_ccl = torch.nn.ReLU()

        self.linear4_ccl = torch.nn.Linear(int(self.cll_input_layer_size * 0.125),
                                           int(self.cll_input_layer_size * 0.0625))
        self.activation4_ccl = torch.nn.ReLU()
        self.linear1_concat = torch.nn.Linear(
            int(self.cll_input_layer_size * 0.0625) + last_layer, 1000)
        self.activation1_concat = torch.nn.ReLU()

        self.linear2_concat = torch.nn.Linear(1000, 500)
        self.activation2_concat = torch.nn.ReLU()

        self.linear3_concat = torch.nn.Linear(500, 250)
        self.activation3_concat = torch.nn.ReLU()

        self.linear4_concat = torch.nn.Linear(250, 1)

    def forward(self, train_cll, train_drug):
        x, edge_index = train_drug.x, train_drug.edge_index
        x = self.conv1_drug(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = self.conv2_drug(x, edge_index)
        x = tg.nn.global_mean_pool(x, train_drug.batch)
        x = self.linear_drug(x)

        train_cll = self.linear1_ccl(train_cll)
        train_cll = self.activation1_ccl(train_cll)
        train_cll = self.linear2_ccl(train_cll)
        train_cll = self.activation2_ccl(train_cll)
        train_cll = self.linear3_ccl(train_cll)
        train_cll = self.activation3_ccl(train_cll)
        train_cll = self.linear4_ccl(train_cll)
        train_cll = self.activation4_ccl(train_cll)

        concat_data = torch.cat((train_cll, x), 1)
        concat_data = self.linear1_concat(concat_data)
        concat_data = self.activation1_concat(concat_data)
        concat_data = self.linear2_concat(concat_data)
        concat_data = self.activation2_concat(concat_data)
        concat_data = self.linear3_concat(concat_data)
        concat_data = self.activation3_concat(concat_data)
        concat_data = self.linear4_concat(concat_data)

        return concat_data
