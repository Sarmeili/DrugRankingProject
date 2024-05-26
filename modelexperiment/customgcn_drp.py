import torch
import torch_geometric as tg
import json
from torch_geometric.nn import MessagePassing, global_mean_pool


class CustomGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_attr_size):
        super(CustomGCNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels + edge_attr_size, out_channels)
        self.edge_mlp = torch.nn.Linear(edge_attr_size, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # Add self-loops to the adjacency matrix.
        edge_index, _ = tg.utils.add_self_loops(edge_index, num_nodes=x.size(0))

        # Add a dummy edge attribute for self-loops.
        loop_attr = torch.zeros(x.size(0), edge_attr.size(1), device=edge_attr.device)
        edge_attr = torch.cat([edge_attr, loop_attr], dim=0)

        # Start propagating messages.
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # x_j has shape [E, in_channels]
        # edge_attr has shape [E, edge_attr_size]
        tmp = torch.cat([x_j, edge_attr], dim=-1)
        return self.lin(tmp)

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        return aggr_out


class DrugRank(torch.nn.Module):
    def __init__(self, cll_size, mol_size, edge_attr_size):
        super(DrugRank, self).__init__()
        with open('config.json') as config_file:
            config = json.load(config_file)

        self.conv1_mol = CustomGCNConv(mol_size, 1000, edge_attr_size)
        self.conv2_mol = CustomGCNConv(1000, 700, edge_attr_size)
        self.conv3_mol = CustomGCNConv(700, 300, edge_attr_size)
        self.conv4_mol = CustomGCNConv(300, 300, edge_attr_size)

        self.linear1_cll = torch.nn.Linear(cll_size, 2000)
        self.linear2_cll = torch.nn.Linear(2000, 2000)
        self.linear3_cll = torch.nn.Linear(2000, 2000)
        self.linear4_cll = torch.nn.Linear(2000, 1000)
        self.linear5_cll = torch.nn.Linear(1000, 1000)
        self.linear6_cll = torch.nn.Linear(1000, 500)
        self.linear7_cll = torch.nn.Linear(500, 300)
        self.linear8_cll = torch.nn.Linear(300, 300)

        self.linear1_comb = torch.nn.Linear(600, 200)
        self.linear2_comb = torch.nn.Linear(200, 100)
        self.linear3_comb = torch.nn.Linear(100, 1)

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
        x_mol = torch.nn.functional.relu(x_mol)
        x_mol = global_mean_pool(x_mol, mol.batch)

        x_cat = torch.cat((x_mol, cll), 1)
        x_cat = self.linear1_comb(x_cat)
        x_cat = torch.nn.functional.relu(x_cat)
        x_cat = self.dropout(x_cat)
        x_cat = self.linear2_comb(x_cat)
        x_cat = torch.nn.functional.relu(x_cat)
        x_cat = self.dropout(x_cat)
        x_cat = self.linear3_comb(x_cat)

        return x_cat