import json
import torch
import torch_geometric as tg


class NetProp:

    def __init__(self):
        with open('config.json') as config_file:
            config = json.load(config_file)
        self.alpha = config['datahandler']['netprop']['alpha']
        self.num_selec = config['datahandler']['netprop']['number_of_selection']

    def netpropagete(self, graph):

        A = tg.utils.to_dense_adj(graph.edge_index)[0]
        w0 = graph.x.reshape(1, -1)[0]
        wt = graph.x.reshape(1, -1)[0]
        D = torch.diag(torch.sum(A, dim=1) ** (-0.5))
        D[D == float("Inf")] = 1
        Aprime = D @ A @ D
        for i in range(10):
            wt0 = wt
            wt = self.alpha * wt @ Aprime + (1 - self.alpha) * w0
            sort, index = torch.sort(wt, 0)
            diff = wt - wt0
            if diff.mean() < 10 ^ (-6):
                break
        return torch.flip(index, [0])

    def create_reduced_graph(self, graph, ppi_df):
        sorted_index = self.netpropagete(graph)


