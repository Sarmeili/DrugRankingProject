import torch
import torch_geometric as tg
import json

class DrugRank(torch.nn.Module):
    def __init__(self, cll_size, mol_size, edge_size):
        super(DrugRank, self).__init__()
        with open('config.json') as config_file:
            config = json.load(config_file)

        # Define the networks for edge conditioning
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(edge_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, mol_size * 1000)
        )
        nn2 = torch.nn.Sequential(
            torch.nn.Linear(edge_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1000 * 700)
        )
        nn3 = torch.nn.Sequential(
            torch.nn.Linear(edge_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 700 * 300)
        )
        nn4 = torch.nn.Sequential(
            torch.nn.Linear(edge_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 300 * 100)
        )

        self.conv1_mol = tg.nn.NNConv(mol_size, 1000, nn1, aggr='mean')
        self.conv2_mol = tg.nn.NNConv(1000, 700, nn2, aggr='mean')
        self.conv3_mol = tg.nn.NNConv(700, 300, nn3, aggr='mean')
        self.conv4_mol = tg.nn.NNConv(300, 100, nn4, aggr='mean')

        self.linear1_cll = torch.nn.Linear(cll_size, 2000)
        self.linear2_cll = torch.nn.Linear(2000, 2000)
        self.linear3_cll = torch.nn.Linear(2000, 2000)
        self.linear4_cll = torch.nn.Linear(2000, 1000)
        self.linear5_cll = torch.nn.Linear(1000, 1000)
        self.linear6_cll = torch.nn.Linear(1000, 900)
        self.linear7_cll = torch.nn.Linear(900, 800)
        self.linear8_cll = torch.nn.Linear(800, 400)

        self.linear1_comb = torch.nn.Linear(500, 200)  # Adjust input size for concatenated features
        self.linear2_comb = torch.nn.Linear(200, 100)
        self.linear3_comb = torch.nn.Linear(100, 1)

        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, cll, mol):
        x_mol, edge_mol, attr_mol = mol.x, mol.edge_index, mol.edge_attr

        # Cell line data processing
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

        # Molecular data processing with NNConv
        x_mol = self.conv1_mol(x_mol, edge_mol, attr_mol)
        x_mol = torch.nn.functional.relu(x_mol)
        x_mol = self.dropout(x_mol)
        x_mol = self.conv2_mol(x_mol, edge_mol, attr_mol)
        x_mol = torch.nn.functional.relu(x_mol)
        x_mol = self.dropout(x_mol)
        x_mol = self.conv3_mol(x_mol, edge_mol, attr_mol)
        x_mol = torch.nn.functional.relu(x_mol)
        x_mol = self.dropout(x_mol)
        x_mol = self.conv4_mol(x_mol, edge_mol, attr_mol)
        x_mol = tg.nn.global_mean_pool(x_mol, mol.batch)

        # Combine cell line and molecular features
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