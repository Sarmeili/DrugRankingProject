import torch
import torch_geometric as tg
import json


class DrugRank(torch.nn.Module):
    def __init__(self, cll_size, mol_size, bio_size):
        super(DrugRank, self).__init__()
        with open('config.json') as config_file:
            config = json.load(config_file)
        self.mol_ll = config['model_experiments']['drugrank']['mol_ll']
        self.bio_ll = config['model_experiments']['drugrank']['bio_ll']
        self.cll_ll = config['model_experiments']['drugrank']['cll_ll']

        self.conv1_mol = tg.nn.GraphConv(mol_size, 200)
        self.conv2_mol = tg.nn.GraphConv(200, 200)
        self.linear_mol = tg.nn.Linear(200, self.mol_ll)

        self.conv1_bio = tg.nn.GCNConv(bio_size, 200)
        self.conv2_bio = tg.nn.GCNConv(200, 200)
        self.linear_bio = tg.nn.Linear(200, self.bio_ll)

        self.conv1_cll = tg.nn.GCNConv(cll_size, 200)
        self.conv2_cll = tg.nn.GCNConv(200, 200)
        self.conv3_cll = tg.nn.GCNConv(200, 200)
        self.conv4_cll = tg.nn.GCNConv(200, 3)
        self.linear1_cll = tg.nn.Linear(3 * 3451, 1000)
        self.linear2_cll = tg.nn.Linear(1000, 1000)
        self.linear3_cll = tg.nn.Linear(1000, self.cll_ll)

        self.linear_drug_1 = tg.nn.Linear(self.mol_ll + self.bio_ll, 500)
        self.linear_drug_2 = tg.nn.Linear(500, self.mol_ll + self.bio_ll)

        self.linear1_concat = tg.nn.Linear(self.mol_ll + self.bio_ll + self.cll_ll, 1000)
        self.linear2_concat = tg.nn.Linear(1000, 1)

        # self.W = torch.nn.Parameter(torch.randn(self.cll_ll, self.bio_ll + self.mol_ll))
        # self.bias = torch.nn.Parameter(torch.randn(1))

    def forward(self, cll, mol, bio):

        x_cll, edge_cll = cll.x, cll.edge_index
        x_mol, edge_mol, attr_mol = mol.x, mol.edge_index, mol.edge_attr
        x_bio, edge_bio = bio.x, bio.edge_index

        x_cll = self.conv1_cll(x_cll, edge_cll)
        x_cll = torch.nn.functional.relu(x_cll)
        x_cll = self.conv2_cll(x_cll, edge_cll)
        x_cll = torch.nn.functional.relu(x_cll)
        x_cll = self.conv3_cll(x_cll, edge_cll)
        x_cll = torch.nn.functional.relu(x_cll)
        x_cll = self.conv4_cll(x_cll, edge_cll)
        x_cll = torch.nn.functional.relu(x_cll)
        # x_cll = tg.nn.global_mean_pool(x_cll, cll.batch)
        x_cll = x_cll.reshape(1, -1)
        x_cll = self.linear1_cll(x_cll)
        x_cll = torch.nn.functional.relu(x_cll)
        x_cll = self.linear2_cll(x_cll)
        x_cll = torch.nn.functional.relu(x_cll)
        x_cll = self.linear3_cll(x_cll)
        x_cll = torch.nn.functional.relu(x_cll)

        x_mol = self.conv1_mol(x_mol, edge_mol)
        x_mol = torch.nn.functional.relu(x_mol)
        x_mol = self.conv2_mol(x_mol, edge_mol)
        x_mol = torch.nn.functional.relu(x_mol)
        x_mol = tg.nn.global_mean_pool(x_mol, mol.batch)
        x_mol = self.linear_mol(x_mol)
        x_mol = torch.nn.functional.relu(x_mol)

        x_bio = self.conv1_bio(x_bio, edge_bio)
        x_bio = torch.nn.functional.relu(x_bio)
        x_bio = self.conv2_bio(x_bio, edge_bio)
        x_bio = torch.nn.functional.relu(x_bio)
        x_bio = x_bio[-1].reshape((1, -1))
        x_bio = self.linear_bio(x_bio)
        x_bio = torch.nn.functional.relu(x_bio)

        x_drug = torch.cat((x_mol, x_bio), 1)
        x_drug = self.linear_drug_1(x_drug)
        x_drug = torch.nn.functional.relu(x_drug)
        x_drug = self.linear_drug_2(x_drug)
        x_drug = torch.nn.functional.relu(x_drug)

        x_cat = torch.cat((x_drug, x_cll), 1)
        x_cat = self.linear1_concat(x_cat)
        x_cat = torch.nn.functional.relu(x_cat)
        x_cat = self.linear2_concat(x_cat)
        x_cat = torch.nn.functional.relu(x_cat)

        # scores = torch.matmul(x_cll @ self.W, x_drug.t()) + self.bias
        return x_cat
