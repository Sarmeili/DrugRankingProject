import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader


class GCNEncoder(torch.nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, 500)
        self.conv2 = GCNConv(1000, 1000)
        self.conv3 = GCNConv(1000, 1000)
        self.conv4 = GCNConv(1000, 1000)
        self.conv5 = GCNConv(1000, 1000)
        self.conv6 = GCNConv(1000, 1000)
        self.conv7 = GCNConv(1000, 1000)
        self.conv7 = GCNConv(1000, embedding_dim)

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.relu(self.conv3(x, edge_index, edge_weight))
        x = F.relu(self.conv4(x, edge_index, edge_weight))
        x = F.relu(self.conv5(x, edge_index, edge_weight))
        x = F.relu(self.conv6(x, edge_index, edge_weight))
        x = self.conv7(x, edge_index, edge_weight)
        return x


class GCNDecoder(torch.nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super(GCNDecoder, self).__init__()
        self.conv1 = GCNConv(embedding_dim, 100)
        self.conv2 = GCNConv(1000, 1000)
        self.conv3 = GCNConv(1000, 1000)
        self.conv4 = GCNConv(1000, 1000)
        self.conv5 = GCNConv(1000, 1000)
        self.conv6 = GCNConv(1000, 1000)
        self.conv7 = GCNConv(1000, 1000)
        self.conv7 = GCNConv(500, output_dim)

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.relu(self.conv3(x, edge_index, edge_weight))
        x = F.relu(self.conv4(x, edge_index, edge_weight))
        x = F.relu(self.conv5(x, edge_index, edge_weight))
        x = F.relu(self.conv6(x, edge_index, edge_weight))
        x = self.conv7(x, edge_index, edge_weight)
        return x


class GCNAutoencoder(torch.nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(GCNAutoencoder, self).__init__()
        self.encoder = GCNEncoder(input_dim, embedding_dim)
        self.decoder = GCNDecoder(embedding_dim, input_dim)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.encoder(x, edge_index, edge_weight)
        h = global_mean_pool(x, data.batch)  # Pooling to create graph-level representation
        x = self.decoder(h, edge_index, edge_weight)
        return x, h