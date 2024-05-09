import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg


class MolGraphAutoencoder(nn.Module):
    def __init__(self, input_dim, encoded_dim):
        super(MolGraphAutoencoder, self).__init__()
        # Encoder
        self.conv1_mol = tg.nn.GraphConv(input_dim, 1000)
        self.conv2_mol = tg.nn.GraphConv(1000, 1000)
        self.conv3_mol = tg.nn.GraphConv(1000, encoded_dim)
        # Decoder
        self.conv4_mol = tg.nn.GraphConv(encoded_dim, 1000)
        self.conv5_mol = tg.nn.GraphConv(1000, 1000)
        self.conv6_mol = tg.nn.GraphConv(1000, input_dim)

        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, mol):
        # Encode
        x_mol, edge_mol, attr_mol = mol.x, mol.edge_index, mol.edge_attr

        x_mol = self.conv1_mol(x_mol, edge_mol)
        x_mol = torch.nn.functional.relu(x_mol)
        x_mol = self.dropout(x_mol)
        x_mol = self.conv2_mol(x_mol, edge_mol)
        x_mol = torch.nn.functional.relu(x_mol)
        x_mol = self.dropout(x_mol)
        encoded = tg.nn.global_mean_pool(x_mol, mol.batch)

        x_mol = self.conv3_mol(x_mol, edge_mol)
        x_mol = torch.nn.functional.relu(x_mol)
        x_mol = self.dropout(x_mol)
        x_mol = self.conv4_mol(x_mol, edge_mol)

        return x_mol, encoded

class CllGraphAutoencoder(nn.Module):
    def __init__(self, input_dim, encoded_dim):
        super(CllGraphAutoencoder, self).__init__()
        # Encoder
        self.conv1_cll = tg.nn.GraphConv(input_dim, 2000)
        self.conv2_cll = tg.nn.GraphConv(2000, 2000)
        self.conv3_cll = tg.nn.GraphConv(2000, encoded_dim)
        # Decoder
        self.conv4_cll = tg.nn.GraphConv(encoded_dim, 2000)
        self.conv5_cll = tg.nn.GraphConv(2000, 2000)
        self.conv6_cll = tg.nn.GraphConv(2000, input_dim)

        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, cll):
        # Encode
        x_cll, edge_cll, attr_cll = cll.x, cll.edge_index, cll.edge_attr
        edge_cll = edge_cll.to(torch.int64)
        x_cll = x_cll.to(torch.float32)
        x_cll = self.conv1_cll(x_cll, edge_cll)
        x_cll = torch.nn.functional.relu(x_cll)
        x_cll = self.dropout(x_cll)
        x_cll = self.conv2_cll(x_cll, edge_cll)
        x_cll = torch.nn.functional.relu(x_cll)
        x_cll = self.dropout(x_cll)
        x_cll = self.conv3_cll(x_cll, edge_cll)
        x_cll = torch.nn.functional.relu(x_cll)
        x_cll = self.dropout(x_cll)
        encoded = tg.nn.global_mean_pool(x_cll, cll.batch)

        x_cll = self.conv4_cll(x_cll, edge_cll)
        x_cll = torch.nn.functional.relu(x_cll)
        x_cll = self.dropout(x_cll)
        x_cll = self.conv5_cll(x_cll, edge_cll)
        x_cll = torch.nn.functional.relu(x_cll)
        x_cll = self.dropout(x_cll)
        x_cll = self.conv6_cll(x_cll, edge_cll)

        return x_cll, encoded


class LinearAutoencoder(nn.Module):
    def __init__(self, input_dim, encoded_dim):
        super(LinearAutoencoder, self).__init__()
        self.linear1_cll = tg.nn.Linear(input_dim, 2000)
        self.linear2_cll = tg.nn.Linear(2000, 1000)
        self.linear3_cll = tg.nn.Linear(1000, 500)
        self.linear4_cll = tg.nn.Linear(500, encoded_dim)

        self.linear5_cll = tg.nn.Linear(encoded_dim, 500)
        self.linear6_cll = tg.nn.Linear(500, 1000)
        self.linear7_cll = tg.nn.Linear(1000, 2000)
        self.linear8_cll = tg.nn.Linear(2000, input_dim)

        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):

        x = self.linear1_cll(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        x = self.linear2_cll(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        x = self.linear3_cll(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        x = self.linear4_cll(x)
        encoded = x
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)

        x = self.linear5_cll(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        x = self.linear6_cll(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        x = self.linear7_cll(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        x = self.linear8_cll(x)

        return x, encoded