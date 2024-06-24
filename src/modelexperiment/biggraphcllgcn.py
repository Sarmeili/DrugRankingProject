import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GraphSAGE
from torch_geometric.data import Data, DataLoader


class GCNEncoder(torch.nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(GCNEncoder, self).__init__()
        self.conv1 = GraphSAGE(input_dim, 500, num_layers=2)
        self.conv2 = GraphSAGE(500, 1000, num_layers=2)
        self.conv3 = GraphSAGE(1000, 1000, num_layers=2)
        self.conv4 = GraphSAGE(1000, 1000, num_layers=2)
        self.conv5 = GraphSAGE(1000, 1000, num_layers=2)
        self.conv6 = GraphSAGE(1000, 1000, num_layers=2)
        self.conv7 = GraphSAGE(1000, 1000, num_layers=2)
        self.conv8 = GraphSAGE(1000, 1000, num_layers=2)
        self.conv9 = GraphSAGE(1000, 1000, num_layers=2)
        self.conv10 = GraphSAGE(1000, 1000, num_layers=2)
        self.conv11 = GraphSAGE(1000, 500, num_layers=2)
        self.conv12 = GraphSAGE(500, embedding_dim, num_layers=2)

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.relu(self.conv3(x, edge_index, edge_weight))
        x = F.relu(self.conv4(x, edge_index, edge_weight))
        x = F.relu(self.conv5(x, edge_index, edge_weight))
        x = F.relu(self.conv6(x, edge_index, edge_weight))
        x = F.relu(self.conv7(x, edge_index, edge_weight))
        x = F.relu(self.conv8(x, edge_index, edge_weight))
        x = F.relu(self.conv9(x, edge_index, edge_weight))
        x = F.relu(self.conv10(x, edge_index, edge_weight))
        x = F.relu(self.conv11(x, edge_index, edge_weight))
        x = self.conv12(x, edge_index, edge_weight)
        return x


class GCNDecoder(torch.nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super(GCNDecoder, self).__init__()
        self.conv1 = GraphSAGE(embedding_dim, 500, num_layers=2)
        self.conv2 = GraphSAGE(500, 1000, num_layers=2)
        self.conv3 = GraphSAGE(1000, 1000, num_layers=2)
        self.conv4 = GraphSAGE(1000, 1000, num_layers=2)
        self.conv5 = GraphSAGE(1000, 1000, num_layers=2)
        self.conv6 = GraphSAGE(1000, 1000, num_layers=2)
        self.conv7 = GraphSAGE(1000, 1000, num_layers=2)
        self.conv8 = GraphSAGE(1000, 1000, num_layers=2)
        self.conv9 = GraphSAGE(1000, 1000, num_layers=2)
        self.conv10 = GraphSAGE(1000, 1000, num_layers=2)
        self.conv11 = GraphSAGE(1000, 500, num_layers=2)
        self.conv12 = GraphSAGE(500, output_dim, num_layers=2)

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.relu(self.conv3(x, edge_index, edge_weight))
        x = F.relu(self.conv4(x, edge_index, edge_weight))
        x = F.relu(self.conv5(x, edge_index, edge_weight))
        x = F.relu(self.conv6(x, edge_index, edge_weight))
        x = F.relu(self.conv7(x, edge_index, edge_weight))
        x = F.relu(self.conv8(x, edge_index, edge_weight))
        x = F.relu(self.conv9(x, edge_index, edge_weight))
        x = F.relu(self.conv10(x, edge_index, edge_weight))
        x = F.relu(self.conv11(x, edge_index, edge_weight))
        x = self.conv12(x, edge_index, edge_weight)
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
        x = self.decoder(x, edge_index, edge_weight)
        return x, h