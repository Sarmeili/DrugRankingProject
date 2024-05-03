import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import torch
import numpy as np
from torch.utils.data import Dataset
import json
from scipy.stats import zscore
import torch_geometric as tg
import random
from tqdm import tqdm
from torch_geometric.data import Data
import rdkit
from datahandler.netprop import NetProp
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.data import DataLoader


class CTRPHandler:
    def __init__(self):
        """
        Data handling class. It is used after data is wrangled and dimensionality reduction is done
        by network propagation.
        """
        with open('config.json') as config_file:
            config = json.load(config_file)
        self.top_k_netprop = config['network_propagation']['top_k']
        self.batch_size = config['datahandler']['ctrp_handler']['batch_size']
        self.use_netprop = config['datahandler']['ctrp_handler']['use_netprop']
        self.device = config['main']['device']

        self.propagated_graph_df = None
        self.read_propagated_graph_df()

        self.response_df = None
        self.read_response_df()

        self.exp_cll_df = None
        self.read_exp_df()

        self.ppi_index_df = None
        self.read_ppi_index_df()

        self.drug_target_df = None
        self.read_drug_target_df()

        self.cmpd_df = None
        self.read_cmpd_df()

    def read_exp_df(self):
        """
        Read cell line gene expression data
        """
        self.exp_cll_df = pd.read_csv('data/wrangled/ccle_exp.csv', index_col=0)
        if self.use_netprop:
            selected_index = list(self.propagated_graph_df['0'])
            self.exp_cll_df = self.exp_cll_df.set_index('nana').iloc[:, selected_index]

    def read_cmpd_df(self):
        """
        Read compound data
        """
        self.cmpd_df = pd.read_csv('data/wrangled/cmpd.csv', index_col=0)

    def read_response_df(self, reverse=True):
        """
        Read response data. Because AUC and IC50 have negative relationship with the response score, all of them got
        negative and then the maximum of response in terms of absolute number has been added to all data.
        :param reverse:
        """
        self.response_df = pd.read_csv('data/wrangled/ctrp.csv')
        max_auc = self.response_df['area_under_curve'].max()
        if reverse:
            self.response_df['area_under_curve'] = self.response_df['area_under_curve'].apply(lambda x: (x*-1)+max_auc)

    def read_propagated_graph_df(self):
        """
        Read selected node data after network propagation has been done. The top_k can be entered through config.json
        file. top_k can be 20, 30, 40, 50
        """
        self.propagated_graph_df = pd.read_csv('data/netprop/top_'+str(self.top_k_netprop)+'_chosen_drug.csv',
                                               index_col=0).astype(int)

    def get_reg_y(self):
        y = self.response_df['area_under_curve'].values
        return y

    def load_y(self, y):
        return DataLoader(y, batch_size=self.batch_size)

    def get_cll_x(self):
        cll_feat = torch.tensor(self.exp_cll_df.reindex(self.response_df['DepMap_ID']).values)
        return cll_feat

    def load_cll(self, cll_feat):
        return DataLoader(cll_feat, batch_size=self.batch_size)

    def get_cmpd_x(self):
        cpd_ids = self.response_df['master_cpd_id'].unique()
        data = []
        for id in cpd_ids:
            data.append({'ID': id,
                        'Graph': self.create_mol_graph(id)})
        df = pd.DataFrame(data)
        df = df.set_index('ID').reindex(self.response_df['master_cpd_id'])
        x = df['Graph'].values
        return x

    def load_cmpd(self, cmpd_feat):
        return DataLoader(cmpd_feat, batch_size=self.batch_size)

    def select_gene_feature(self):
        """
        Function that use dimensionality reduction data to select genes and transform graphs to the reduced version.
        All the ppi edges, drug targets and edge features are reindexed in this function. The output of this function
        is then used when ppi of cell line and drug target interaction in a graph is created.
        :return:
        selected_gene_df: that contains only selected genes from network propageation
        edges: updated index of each pair of nodes known as edges
        edges_attrib: feature of each edge
        """
        self.read_exp_df()
        self.read_propagated_graph_df()
        self.read_drug_target_df()
        self.read_ppi_index_df()
        self.drug_target_df['new_index'] = None
        selected_index = list(self.propagated_graph_df['0'])
        selected_gene_df = self.exp_cll_df.set_index('nana').iloc[:, selected_index]
        edges = self.ppi_index_df[['protein1', 'protein2']].to_numpy().reshape(2, -1)
        edges_attrib = self.ppi_index_df['combined_score'].to_numpy().reshape(-1, 1)
        edges = edges.reshape(-1, 2)
        mask = np.isin(edges, selected_index).all(where=[[True, True]], axis=1)
        edges = edges[mask]
        edges = edges.reshape(2, -1)
        edges_attrib = edges_attrib[mask]
        for i in range(len(selected_index)):
            edges[0][edges[0] == selected_index[i]] = i
            edges[1][edges[1] == selected_index[i]] = i
            if selected_index[i] in list(self.drug_target_df['index_target']):
                target_index = self.drug_target_df[self.drug_target_df['index_target'] == selected_index[i]].index
                self.drug_target_df.loc[target_index, 'new_index'] = i
        return selected_gene_df.reset_index(), edges, edges_attrib

    def get_npgenes_drugs_train(self):
        self.read_exp_df()
        self.read_propagated_graph_df()
        self.read_drug_target_df()
        self.read_ppi_index_df()
        self.read_response_df()
        response_df = self.response_df[:int(0.8*len(self.response_df))]
        n_samples = len(response_df)
        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            response_df = self.response_df[start:end]
            cpd_ids = response_df['master_cpd_id']
            response = torch.tensor(response_df['area_under_curve'].values)
            self.drug_target_df['new_index'] = None
            selected_index = list(self.propagated_graph_df['0'])
            selected_gene_df = self.exp_cll_df.set_index('nana').iloc[:, selected_index]
            cll_feat = torch.tensor(selected_gene_df.reindex(response_df['DepMap_ID']).values)
            cpd_graphs = []
            for id in cpd_ids:
                graph = self.create_mol_graph(id)
                cpd_graphs.append(graph)
                # print(graph)
            yield cll_feat, cpd_graphs, response

    def get_npgenes_drugs_val(self):
        self.read_exp_df()
        self.read_propagated_graph_df()
        self.read_drug_target_df()
        self.read_ppi_index_df()
        self.read_response_df()
        response_df = self.response_df[int(0.8*len(self.response_df)):]
        n_samples = len(response_df)
        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            response_df = self.response_df[start:end]
            cpd_ids = response_df['master_cpd_id']
            response = torch.tensor(response_df['area_under_curve'].values)
            self.drug_target_df['new_index'] = None
            selected_index = list(self.propagated_graph_df['0'])
            selected_gene_df = self.exp_cll_df.set_index('nana').iloc[:, selected_index]
            cll_feat = torch.tensor(selected_gene_df.reindex(response_df['DepMap_ID']).values)
            cpd_graphs = []
            for id in cpd_ids:
                graph = self.create_mol_graph(id)
                cpd_graphs.append(graph)
                # print(graph)
            yield cll_feat, cpd_graphs, response

    def listwise_ranking_df(self):
        """
        Makes all ranking dataframe to 3 dataframes: training set of 80% , validation set of 10%, test set of 10% of
        dataset.
        :return:
        train set
        validation set
        test set
        """
        self.read_response_df()
        cll_list = list(self.response_df['DepMap_ID'])
        cll_list = list(set(sorted(cll_list)))
        cd_df = self.response_df.set_index(['DepMap_ID', 'master_cpd_id'])
        drug_list = []
        response_list = []
        for cll in cll_list:
            drug_list.append(list(cd_df.loc[cll].index))
            response_list.append(list(cd_df.loc[cll]['area_under_curve']))
        response_ranking_df = pd.DataFrame({'cll': cll_list,
                                            'drug': drug_list,
                                            'response': response_list})
        response_ranking_train = response_ranking_df[:int(len(response_ranking_df)*0.8)]
        response_ranking_val = response_ranking_df[
                               int(len(response_ranking_df) * 0.8):int(len(response_ranking_df) * 0.9)]
        response_ranking_test = response_ranking_df[
                               int(len(response_ranking_df) * 0.9):]
        del cll_list
        del cd_df
        del drug_list
        del response_list
        return response_ranking_train, response_ranking_val, response_ranking_test

    def generate_cll_drug_response(self, response_ranking_df):
        """
        A generator to access to each row of our dataset for further creation of graphs and processing. Batch size can
        be designated in config file.
        :param response_ranking_df: Each train, val or test dataframe
        :return:
        cell lines code, list of drugs code and corresponding list of responses of each cell line and drugs
        """
        clls = list(response_ranking_df['cll'])
        drugs = list(response_ranking_df['drug'])
        responses = list(response_ranking_df['response'])
        n_samples = len(response_ranking_df)
        indices = np.arange(n_samples)
        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            yield clls[start:end], drugs[start:end], responses[start:end]

    def netprop_dim_reduction(self):
        """
           Propagate PPI network with regard to drugs targets in order to find
           top 20, top 30, top 40 and top 50 genes in the big graph as a way
           to reduce dimension.
           csv files can be found in data folder
        """
        self.read_drug_target_df()
        self.read_ppi_index_df()
        self.read_exp_df()
        selected_index_20 = []
        selected_index_30 = []
        selected_index_40 = []
        selected_index_50 = []
        netprop = NetProp()
        genes = self.exp_cll_df.columns[1:]
        drugs = self.drug_target_df['master_cpd_id'].unique()
        for drug in tqdm(drugs, total=len(drugs)):
            targets = list(self.drug_target_df[self.drug_target_df['master_cpd_id'] == drug]['index_target'])
            x = [0.00001 for i in range(len(genes))]
            for target in targets:
                x[target] = 1.0
            x = np.array(x).reshape((-1, 1))
            edges = self.ppi_index_df[['protein1', 'protein2']].to_numpy().reshape(2, -1)
            edges_attrib = self.ppi_index_df['combined_score'].to_numpy().reshape(-1, 1)
            bio_graph = tg.data.Data(x=torch.from_numpy(x).to(torch.float32), edge_index=torch.from_numpy(edges),
                                     edges_attrib=edges_attrib)
            wt = netprop.netpropagete(bio_graph)

            selected_index_20 += wt[:20].tolist()
            selected_index_20 = list(set(selected_index_20))

            selected_index_30 += wt[:30].tolist()
            selected_index_30 = list(set(selected_index_30))

            selected_index_40 += wt[:40].tolist()
            selected_index_40 = list(set(selected_index_40))

            selected_index_50 += wt[:50].tolist()
            selected_index_50 = list(set(selected_index_50))

        drug_chosen = pd.Series(selected_index_20)
        drug_chosen.to_csv('data/netprop/top_20_chosen_drug.csv')
        drug_chosen = pd.Series(selected_index_30)
        drug_chosen.to_csv('data/netprop/top_30_chosen_drug.csv')
        drug_chosen = pd.Series(selected_index_40)
        drug_chosen.to_csv('data/netprop/top_40_chosen_drug.csv')
        drug_chosen = pd.Series(selected_index_50)
        drug_chosen.to_csv('data/netprop/top_50_chosen_drug.csv')

    def create_mol_graph(self, drug_code):
        """
        Create the graph of molecule with each atoms' and edges' features. The list of features for each can be seen at
        lists below.
        :param drug_code: code for a compound in CTRP dataset
        :return: Torch Geometric Graph to be given to models.
        """
        self.read_cmpd_df()
        cpd_df = self.cmpd_df.set_index('master_cpd_id')
        cpd_df = cpd_df.reindex([drug_code])
        smile = list(cpd_df['cpd_smiles'])[0]
        mol = Chem.MolFromSmiles(smile)
        atoms = mol.GetAtoms()
        edges = mol.GetBonds()
        atoms_feature = []
        edges_list = [[], []]
        edges_feature = []
        for atom in atoms:
            atom_feature = [atom.GetAtomicNum(), atom.GetDegree(), atom.GetTotalNumHs(), atom.GetTotalValence(),
                            atom.GetNumRadicalElectrons(), atom.GetFormalCharge(), atom.GetIsAromatic(), atom.GetMass(),
                            atom.GetIsotope(), atom.InvertChirality()]
            chiral_tags = [rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                           rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                           rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                           rdkit.Chem.rdchem.ChiralType.CHI_OTHER,
                           rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL,
                           rdkit.Chem.rdchem.ChiralType.CHI_ALLENE,
                           rdkit.Chem.rdchem.ChiralType.CHI_SQUAREPLANAR,
                           rdkit.Chem.rdchem.ChiralType.CHI_TRIGONALBIPYRAMIDAL,
                           rdkit.Chem.rdchem.ChiralType.CHI_OCTAHEDRAL]
            for i in range(len(chiral_tags)):
                if atom.GetChiralTag() == chiral_tags[i]:
                    atom_feature.append(1)
                else:
                    atom_feature.append(0)
            hybride_type = [rdkit.Chem.rdchem.HybridizationType.S,
                            rdkit.Chem.rdchem.HybridizationType.SP,
                            rdkit.Chem.rdchem.HybridizationType.SP2,
                            rdkit.Chem.rdchem.HybridizationType.SP3,
                            rdkit.Chem.rdchem.HybridizationType.SP2D,
                            rdkit.Chem.rdchem.HybridizationType.SP3D,
                            rdkit.Chem.rdchem.HybridizationType.SP3D2,
                            rdkit.Chem.rdchem.HybridizationType.OTHER]
            for i in range(len(hybride_type)):
                if atom.GetHybridization() == hybride_type[i]:
                    atom_feature.append(1)
                else:
                    atom_feature.append(0)
            atoms_feature.append(atom_feature)

        for edge in edges:
            atom1_idx = edge.GetBeginAtomIdx()
            atom2_idx = edge.GetEndAtomIdx()
            edges_list[0].append(atom1_idx)
            edges_list[0].append(atom2_idx)
            edges_list[1].append(atom2_idx)
            edges_list[1].append(atom1_idx)
            edge_feature = []
            bond_type = [rdkit.Chem.rdchem.BondType.UNSPECIFIED,
                         rdkit.Chem.rdchem.BondType.SINGLE,
                         rdkit.Chem.rdchem.BondType.DOUBLE,
                         rdkit.Chem.rdchem.BondType.TRIPLE,
                         rdkit.Chem.rdchem.BondType.QUADRUPLE,
                         rdkit.Chem.rdchem.BondType.QUINTUPLE,
                         rdkit.Chem.rdchem.BondType.HEXTUPLE,
                         rdkit.Chem.rdchem.BondType.ONEANDAHALF,
                         rdkit.Chem.rdchem.BondType.TWOANDAHALF,
                         rdkit.Chem.rdchem.BondType.THREEANDAHALF,
                         rdkit.Chem.rdchem.BondType.FOURANDAHALF,
                         rdkit.Chem.rdchem.BondType.FIVEANDAHALF,
                         rdkit.Chem.rdchem.BondType.AROMATIC,
                         rdkit.Chem.rdchem.BondType.IONIC,
                         rdkit.Chem.rdchem.BondType.THREECENTER,
                         rdkit.Chem.rdchem.BondType.DATIVEONE,
                         rdkit.Chem.rdchem.BondType.DATIVE,
                         rdkit.Chem.rdchem.BondType.DATIVEL,
                         rdkit.Chem.rdchem.BondType.DATIVER,
                         rdkit.Chem.rdchem.BondType.OTHER,
                         rdkit.Chem.rdchem.BondType.ZERO]
            for i in range(len(bond_type)):
                if edge.GetBondType() == bond_type[i]:
                    edge_feature.append(1)
                else:
                    edge_feature.append(0)
            edge_feature.append(edge.GetBondTypeAsDouble())
            # edge_feature.append(edge.GetValenceContrib())
            edge_feature.append(edge.GetIsAromatic())
            edge_feature.append(edge.GetIsConjugated())
            bond_dir = [rdkit.Chem.rdchem.BondDir.NONE,
                        rdkit.Chem.rdchem.BondDir.BEGINWEDGE,
                        rdkit.Chem.rdchem.BondDir.BEGINDASH,
                        rdkit.Chem.rdchem.BondDir.ENDDOWNRIGHT,
                        rdkit.Chem.rdchem.BondDir.ENDUPRIGHT,
                        rdkit.Chem.rdchem.BondDir.EITHERDOUBLE,
                        rdkit.Chem.rdchem.BondDir.UNKNOWN,
                        rdkit.Chem.rdchem.BondDir.NONE]
            for i in range(len(bond_dir)):
                if edge.GetBondDir() == bond_dir[i]:
                    edge_feature.append(1)
                else:
                    edge_feature.append(0)
            bond_stereo = [rdkit.Chem.rdchem.BondStereo.STEREONONE,
                           rdkit.Chem.rdchem.BondStereo.STEREOANY,
                           rdkit.Chem.rdchem.BondStereo.STEREOZ,
                           rdkit.Chem.rdchem.BondStereo.STEREOE,
                           rdkit.Chem.rdchem.BondStereo.STEREOCIS,
                           rdkit.Chem.rdchem.BondStereo.STEREOTRANS]
            for i in range(len(bond_stereo)):
                if edge.GetStereo() == bond_stereo[i]:
                    edge_feature.append(1)
                else:
                    edge_feature.append(0)
            edges_feature.append(edge_feature)
            edges_feature.append(edge_feature)

        mol_graph = tg.data.Data(x=atoms_feature,
                                 edge_index=torch.tensor(edges_list),
                                 edge_attr=edges_feature)
        mol_graph = mol_graph.to(self.device)
        mol_graph.x = torch.tensor(mol_graph.x, dtype=torch.float32)
        mol_graph.edge_attr = torch.tensor(mol_graph.edge_attr, dtype=torch.float32)

        return mol_graph

    def get_cll_graph_x(self):
        selected_index = list(self.propagated_graph_df['0'])
        edges = self.ppi_index_df[['protein1', 'protein2']].to_numpy().reshape(2, -1)
        edges_attrib = self.ppi_index_df['combined_score'].to_numpy().reshape(-1, 1)
        edges = edges.reshape(-1, 2)
        mask = np.isin(edges, selected_index).all(where=[[True, True]], axis=1)
        edges = edges[mask]
        edges = edges.reshape(2, -1)
        edges_attrib = edges_attrib[mask]
        for i in range(len(selected_index)):
            edges[0][edges[0] == selected_index[i]] = i
            edges[1][edges[1] == selected_index[i]] = i

        x_blank = torch.zeros(len(self.exp_cll_df.iloc[0])).reshape(-1, 1)
        blank_graph = tg.data.Data(x=x_blank, edge_index=torch.from_numpy(edges).to(torch.int32), edge_attr=edges_attrib)
        cll_feat = torch.tensor(self.exp_cll_df.reindex(self.response_df['DepMap_ID']).values)
        graphs = []
        for feats in cll_feat:
            blank_graph.x = feats.reshape(-1, 1).to(torch.int64)
            graphs.append(blank_graph)
        return graphs


    def create_cll_bio_graph(self, drug_code, cll_code, exp_cll_df, edges, edges_attrib):
        """
        Create ppi graph with each node be the expression of gene of that protein. Also Creates same graph but with an
        hypothetical node that represent drug and its interaction with target.
        :param drug_code: code for compound in CTRP dataset
        :param cll_code:code for cell linse in DepMapPortal
        :param exp_cll_df: updated gene expression dataframe of cell lines
        :param edges: pairs of nodes that represent the connection between them
        :param edges_attrib: feature of each edge.
        :return:
        cll_graph : ppi graph
        bio_graph : ppi graph + drug-target
        """
        target = list(self.drug_target_df[self.drug_target_df['master_cpd_id'] == drug_code]['new_index'])
        gene_exp = exp_cll_df[exp_cll_df['nana'] == cll_code]
        x = np.array(list(gene_exp.iloc[0][1:])).reshape((-1, 1))
        cll_graph = tg.data.Data(x=torch.from_numpy(x.astype(np.float32)),
                                 edge_index=torch.from_numpy(edges.astype(np.int32)),
                                 edge_attr=torch.from_numpy(edges_attrib.astype(np.float32)))
        drug_feat = [0.0001]
        drug_index = len(x)
        edges = edges.reshape(-1, 2)
        for target_of_drug in target:
            edges = np.append(edges, [target_of_drug, drug_index])
            edges = np.append(edges, [drug_index, target_of_drug])
            edges_attrib = np.append(edges_attrib, [[edges_attrib.max()]])
            edges_attrib = np.append(edges_attrib, [[edges_attrib.max()]])
        edges = edges.reshape(2, -1)
        x = np.append(x, [[drug_feat]]).reshape(-1, 1)
        bio_graph = tg.data.Data(x=torch.from_numpy(x.astype(np.float32)),
                                 edge_index=torch.from_numpy(edges.astype(np.int32)),
                                 edge_attr=torch.from_numpy(edges_attrib.astype(np.float32).reshape(-1, 1)))
        return cll_graph, bio_graph

    def read_drug_target_df(self):
        """
        Read drug target dataframe
        """
        self.drug_target_df = pd.read_csv('data/wrangled/drug_target.csv')

    def read_ppi_index_df(self):
        """
        Read ppi dataframe
        """
        self.ppi_index_df = pd.read_csv('data/wrangled/ppi.csv')












