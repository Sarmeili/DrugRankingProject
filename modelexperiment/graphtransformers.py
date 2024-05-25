import torch
import torch_geometric as tg
import json


class DrugRank(torch.nn.Module):
    def __init__(self, cll_size, mol_size, edge_size):
        super(DrugRank, self).__init__()
        with open('config.json') as config_file:
            config = json.load(config_file)

        self.tconv1_mol = tg.nn.TransformerConv(mol_size, 1000, heads=4, edge_dim=edge_size)
        self.tconv2_mol = tg.nn.TransformerConv(1000*4, 700, heads=4, edge_dim=edge_size)
        self.tconv3_mol = tg.nn.TransformerConv(700*4, 300, heads=4, edge_dim=edge_size)
        self.tconv4_mol = tg.nn.TransformerConv(300*4, 100, heads=4, edge_dim=edge_size)

        self.linear1_cll = tg.nn.Linear(cll_size, 2000)
        self.linear2_cll = tg.nn.Linear(2000, 2000)
        self.linear3_cll = tg.nn.Linear(2000, 2000)
        self.linear4_cll = tg.nn.Linear(2000, 1000)
        self.linear5_cll = tg.nn.Linear(1000, 1000)
        self.linear6_cll = tg.nn.Linear(1000, 900)
        self.linear7_cll = tg.nn.Linear(900, 800)
        self.linear8_cll = tg.nn.Linear(800, 400)

        # self.bilinear = torch.nn.Bilinear(100, 100, 1)

        self.linear1_comb = tg.nn.Linear(800, 200)
        self.linear2_comb = tg.nn.Linear(200, 100)
        self.linear3_comb = tg.nn.Linear(100, 1)

        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, cll, mol):

        x_mol, edge_mol, attr_mol = mol.x, mol.edge_index, mol.edge_attr
        cll = self.linear1_cll(cll)
        cll = torch.nn.functional.relu(cll)
        cll = self.dropout(cll)
        cll = self.linear2_cll(cll)
        cll = torch.nn.functional.relu(cll)
        cll = self.dropout(cll)
        cll = self.linear3_cll(cll)
        cll = torch.nn.functional.relu(cll)
        cll = self.dropout(cll)
        cll = self.linear4_cll(cll)
        cll = torch.nn.functional.relu(cll)
        cll = self.dropout(cll)
        cll = self.linear5_cll(cll)
        cll = torch.nn.functional.relu(cll)
        cll = self.dropout(cll)
        cll = self.linear6_cll(cll)
        cll = torch.nn.functional.relu(cll)
        cll = self.dropout(cll)
        cll = self.linear7_cll(cll)
        cll = torch.nn.functional.relu(cll)
        cll = self.dropout(cll)
        cll = self.linear8_cll(cll)

        x_mol = self.tconv1_mol(x_mol, edge_mol, attr_mol)
        x_mol = torch.nn.functional.relu(x_mol)
        x_mol = self.dropout(x_mol)
        x_mol = self.tconv2_mol(x_mol, edge_mol, attr_mol)
        x_mol = torch.nn.functional.relu(x_mol)
        x_mol = self.dropout(x_mol)
        x_mol = self.tconv3_mol(x_mol, edge_mol, attr_mol)
        x_mol = torch.nn.functional.relu(x_mol)
        x_mol = self.dropout(x_mol)
        x_mol = self.tconv4_mol(x_mol, edge_mol, attr_mol)
        x_mol = tg.nn.global_mean_pool(x_mol, mol.batch)

        x_cat = torch.cat((x_mol, cll), 1)
        x_cat = self.linear1_comb(x_cat)
        x_cat = torch.nn.functional.relu(x_cat)
        x_cat = self.dropout(x_cat)
        x_cat = self.linear2_comb(x_cat)
        x_cat = torch.nn.functional.relu(x_cat)
        x_cat = self.dropout(x_cat)
        x_cat = self.linear3_comb(x_cat)
        # x_cat = self.bilinear(x_mol, cll)

        return x_cat
