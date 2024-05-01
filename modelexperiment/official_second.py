import torch
import torch_geometric as tg
import json


class DrugRank(torch.nn.Module):
    def __init__(self, cll_size, mol_size):
        super(DrugRank, self).__init__()
        with open('config.json') as config_file:
            config = json.load(config_file)

        self.conv1_mol = tg.nn.GraphConv(mol_size, 500)
        self.conv2_mol = tg.nn.GraphConv(500, 200)
        self.linear_mol = tg.nn.Linear(200, 100)

        self.linear1_cll = tg.nn.Linear(cll_size, 2000)
        self.linear2_cll = tg.nn.Linear(2000, 1000)
        self.linear3_cll = tg.nn.Linear(1000, 500)
        self.linear4_cll = tg.nn.Linear(500, 200)

        self.linear1_comb = tg.nn.Linear(100 + 200, 200)
        self.linear2_comb = tg.nn.Linear(200, 100)
        self.linear3_comb = tg.nn.Linear(100, 1)

        # self.W = torch.nn.Parameter(torch.randn(self.cll_ll, self.bio_ll + self.mol_ll))
        # self.bias = torch.nn.Parameter(torch.randn(1))

    def forward(self, cll, mol):

        x_mol, edge_mol, attr_mol = mol.x, mol.edge_index, mol.edge_attr

        cll = self.linear1_cll(cll)
        cll = torch.nn.functional.relu(cll)
        cll = self.linear2_cll(cll)
        cll = torch.nn.functional.relu(cll)
        cll = self.linear3_cll(cll)
        cll = torch.nn.functional.relu(cll)
        cll = self.linear4_cll(cll)

        x_mol = self.conv1_mol(x_mol, edge_mol)
        x_mol = torch.nn.functional.relu(x_mol)
        x_mol = self.conv2_mol(x_mol, edge_mol)
        x_mol = torch.nn.functional.relu(x_mol)
        x_mol = tg.nn.global_mean_pool(x_mol, mol.batch)
        x_mol = self.linear_mol(x_mol)

        x_cat = torch.cat((x_mol, cll), 1)
        x_cat = self.linear1_comb(x_cat)
        x_cat = torch.nn.functional.relu(x_cat)
        x_cat = self.linear2_comb(x_cat)
        x_cat = torch.nn.functional.relu(x_cat)
        x_cat = self.linear3_comb(x_cat)

        # x_cat = torch.nn.functional.relu(x_cat)

        # scores = torch.matmul(x_cll @ self.W, x_drug.t()) + self.bias
        return x_cat
