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

from torch_geometric.data import Data

from datahandler.netprop import NetProp


class CTRPHandler:

    def __init__(self, data_volume):
        self.ppi_index_df = None
        self.ppi_df = None
        self.drug_target_df = None
        self.data_volume = data_volume
        self.mean = None
        self.std = None
        self.cmpd_df = None
        self.exp_cll_df = None
        self.mut_cll_df = None
        self.response_df = None
        with open('config.json') as config_file:
            config = json.load(config_file)
        self.task = config['task']
        self.drug_feat = config['datahandler']['ctrp_drugranker']['drug_feat']
        self.cll_feat = config['datahandler']['ctrp_drugranker']['cll_feat']
        self.is_pca = config['datahandler']['ctrp_drugranker']['dim_reduction']['pca']['is_pca']
        self.q = config['datahandler']['ctrp_drugranker']['dim_reduction']['pca']['q']
        self.test_percentage = config['datahandler']['ctrp_drugranker']['test_percentage']
        self.source = config['datahandler']['ctrp_drugranker']['source']
        self.last_layer = config['model_experiments']['graphmol_mlp']['last_layer']
        self.k_of_list = config['datahandler']['ctrp_drugranker']['k_of_list']
        self.num_selected = config['datahandler']['netprop']['number_of_selection']

    def z_score_calculation(self, x):
        return (x - self.mean) / self.std

    def read_cll_df(self):
        if "gene_exp" in self.cll_feat:
            self.exp_cll_df = pd.read_csv('data/new_ccle_exp.csv', index_col=0)
        if "gene_mut" in self.cll_feat:
            self.mut_cll_df = pd.read_csv('data/CCLE_mutations_bool_damaging.csv', index_col=0)

    def read_cmpd_df(self):
        if self.source == 'DrugRanker':
            self.cmpd_df = pd.read_csv('data/ctrp_drugranker/cmpd_id_name_group_smiles.txt', sep='\t')

    def read_response_df(self):
        if self.source == 'DrugRanker':
            if self.task == 'regression':
                self.response_df = pd.read_csv('data/ctrp_drugranker/final_list_auc.txt')
                # self.mean = self.response_df['auc'].mean()
                # self.std = self.response_df['auc'].std()
                # self.response_df['auc'] = self.response_df['auc'].apply(self.z_score_calculation)
                self.response_df = self.response_df[int(len(self.response_df) * self.data_volume[0]):int(
                    len(self.response_df) * self.data_volume[1])]
            elif self.task == 'ranking':
                self.response_df = pd.read_csv('data/res.csv')

    def listwise_ranking_df(self):
        self.read_response_df()
        cll_list = list(self.response_df['broadid'])
        cll_list = list(set(cll_list))
        cd_df = self.response_df.set_index(['broadid', 'cpdid'])
        drug_list = []
        response_list = []
        for cll in cll_list:
            drug_list.append(list(cd_df.loc[cll].index))
            response_list.append(list(cd_df.loc[cll]['auc']))
        response_ranking_df = pd.DataFrame({'cll': cll_list,
                                            'drug': drug_list,
                                            'response': response_list})
        response_ranking_df = response_ranking_df[int(len(response_ranking_df) * self.data_volume[0]):int(
            len(response_ranking_df) * self.data_volume[1])]
        del cll_list
        del cd_df
        del drug_list
        del response_list
        return response_ranking_df

    def listwise_drug_molecule_bio(self, response_ranking_df):
        self.read_cmpd_df()
        self.read_drug_target_df()
        self.read_ppi_df()
        self.read_cll_df()
        list_of_graphs_mol = []
        list_of_graphs_bio = []
        netprop = NetProp()
        for i, data in response_ranking_df.iterrows():
            cpd_df = self.cmpd_df.set_index('master_cpd_id')
            cpd_df = cpd_df.reindex(data['drug'])
            graph_of_q_mol = []
            for smile in list(cpd_df['cpd_smiles']):
                mol_graph = tg.utils.from_smiles(smile)
                mol_graph = mol_graph.to('cuda')
                mol_graph.x = torch.tensor(mol_graph.x, dtype=torch.float32)
                mol_graph.edge_attr = torch.tensor(mol_graph.edge_attr, dtype=torch.float32)
                graph_of_q_mol.append(mol_graph)
            del mol_graph
            del cpd_df
            list_of_graphs_mol.append(graph_of_q_mol)
            graph_of_q_bio = []
            for drug in data['drug']:
                target = list(self.drug_target_df[self.drug_target_df['master_cpd_id'] == drug]['index_of_edge'])
                gene_exp = self.exp_cll_df[self.exp_cll_df['nana'] == data['cll']]
                edges = self.ppi_index_df[['protein1', 'protein2']].to_numpy().reshape(2, -1)
                edges_attrib = self.ppi_index_df['combined_score'].to_numpy().reshape(-1, 1)
                #######
                x = [0.00001 for i in range(len(list(gene_exp.iloc[0][1:])))]
                for target_drug in target:
                    x[target_drug] = 1.0
                x = np.array(x).reshape((-1, 1))
                bio_graph = tg.data.Data(x=torch.from_numpy(x).to(torch.float32), edge_index=torch.from_numpy(edges),
                                         edges_attrib=edges_attrib)
                wt = netprop.netpropagete(bio_graph)
                selected_index = wt[:self.num_selected]
                edges = edges.reshape(-1, 2)
                mask = np.isin(edges, selected_index).all(axis=1)
                edges = edges[mask]
                print(len(edges_attrib))
                edges_attrib = edges_attrib[mask]
                print(len(edges_attrib))
                edges = edges.reshape(2, -1)
                for i in range(len(selected_index)):
                    edges[0][edges[0] == selected_index[i].numpy()] = i
                    edges[1][edges[1] == selected_index[i].numpy()] = i
                    target = np.array(target)
                    target[target == selected_index[i].numpy()] = i
                #######
                x = np.array(list(gene_exp.iloc[0][1:])).reshape((-1, 1))[selected_index]
                drug_feat = [0.0001]
                drug_index = len(x)
                for target_of_drug in target:
                    np.append(edges[0], [target_of_drug, drug_index])
                    np.append(edges[1], [drug_index, target_of_drug])
                    np.append(edges_attrib, [[edges_attrib.max()]])
                x = np.append(x, [[drug_feat]])
                del target
                del gene_exp
                del drug_feat
                bio_graph = tg.data.Data(x=torch.from_numpy(x.astype(np.float32)), edge_index=torch.from_numpy(edges.astype(np.int32)), edge_attr=torch.from_numpy(edges_attrib.astype(np.float32)))
                del x
                del edges
                del edges_attrib
                graph_of_q_bio.append(bio_graph)
                break
            list_of_graphs_bio.append(graph_of_q_bio)
            break
        return list_of_graphs_mol, list_of_graphs_bio

    def listwise_drug_molecule(self, response_ranking_df):
        self.read_cmpd_df()
        list_of_graphs = []
        for i, data in response_ranking_df.iterrows():
            cpd_df = self.cmpd_df.set_index('master_cpd_id')
            cpd_df = cpd_df.reindex(data['drug'])
            graph_of_q = []
            for smile in list(cpd_df['cpd_smiles']):
                mol_graph = tg.utils.from_smiles(smile)
                mol_graph = mol_graph.to('cuda')
                mol_graph.x = torch.tensor(mol_graph.x, dtype=torch.float32)
                mol_graph.edge_attr = torch.tensor(mol_graph.edge_attr, dtype=torch.float32)
                graph_of_q.append(mol_graph)
            list_of_graphs.append(graph_of_q)
        del mol_graph
        del cpd_df
        return list_of_graphs

    def listwise_drug_bio(self, response_ranking_df):
        self.read_drug_target_df()
        self.read_ppi_df()
        self.read_cll_df()
        list_of_graphs = []
        for i, data in response_ranking_df.iterrows():
            graph_of_q = []
            for drug in data['drug']:
                target = list(self.drug_target_df[self.drug_target_df['master_cpd_id'] == drug]['index_of_edge'])
                gene_exp = self.exp_cll_df[self.exp_cll_df['nana'] == data['cll']]
                drug_feat = [0.0001]
                x = np.array(list(gene_exp.iloc[0][1:])).reshape((-1, 1))
                drug_index = len(x) + 1
                edges = self.ppi_index_df[['protein1', 'protein2']].to_numpy().reshape(2, -1)
                edges_attrib = self.ppi_index_df['combined_score'].to_numpy().reshape(-1, 1)
                for target_of_drug in target:
                    np.append(edges[0], [target_of_drug, drug_index])
                    np.append(edges[1], [drug_index, target_of_drug])
                    np.append(edges_attrib, [[edges_attrib.max()]])
                np.append(x, [drug_feat])
                bio_graph = tg.data.Data(x=x, edge_index=edges, edges_attrib=edges_attrib)
                graph_of_q.append(bio_graph)
            list_of_graphs.append(graph_of_q)
        return list_of_graphs

    def bio_dimreduction(self, response_ranking_df):
        self.read_drug_target_df()
        self.read_ppi_df()
        self.read_cll_df()
        list_of_graphs = []
        netprop = NetProp()
        for i, data in response_ranking_df.iterrows():
            graph_of_q = []
            for drug in data['drug']:
                target = list(self.drug_target_df[self.drug_target_df['master_cpd_id'] == drug]['index_of_edge'])
                gene_exp = self.exp_cll_df[self.exp_cll_df['nana'] == data['cll']]
                x = [0.00001 for i in range(len(list(gene_exp.iloc[0][1:])))]
                for target_drug in target:
                    x[target_drug] = 1.0
                x = np.array(x).reshape((-1, 1))
                edges = self.ppi_index_df[['protein1', 'protein2']].to_numpy().reshape(2, -1)
                edges_attrib = self.ppi_index_df['combined_score'].to_numpy().reshape(-1, 1)
                bio_graph = tg.data.Data(x=torch.from_numpy(x).to(torch.float32), edge_index=torch.from_numpy(edges), edges_attrib=edges_attrib)
                graph_of_q.append(bio_graph)

            break
        return graph_of_q

            #list_of_graphs.append(graph_of_q)
        #return list_of_graphs

    def read_drug_target_df(self):
        self.drug_target_df = pd.read_csv('data/ctrp_drugranker/drugs_with_target.csv')

    def read_ppi_df(self):
        self.ppi_df = pd.read_csv('data/9606.protein.links.v12.0.txt', sep='\t')
        self.ppi_index_df = pd.read_csv('data/ppi_new_edge.csv')

    def create_tensor_feat_cll(self):
        self.read_cll_df()
        self.read_response_df()
        if 'gene_exp' in self.cll_feat:
            self.exp_cll_df = self.exp_cll_df.astype('float32')
            exp_cll = self.exp_cll_df.reindex(self.response_df['broadid'])
            cll_exp_tensor = torch.from_numpy(exp_cll.to_numpy())
            if self.is_pca:
                pca = torch.pca_lowrank(cll_exp_tensor, q=self.q)
                cll_exp_tensor = pca[0]
        else:
            cll_exp_tensor = None
        if 'gene_mut' in self.cll_feat:
            self.mut_cll_df = self.mut_cll_df.astype('float32')
            mut_cll = self.mut_cll_df.reindex(self.response_df['broadid'])
            cll_mut_tensor = torch.from_numpy(mut_cll.to_numpy())
            if self.is_pca:
                pca = torch.pca_lowrank(cll_mut_tensor, q=self.q)
                cll_mut_tensor = pca[0]
        else:
            cll_mut_tensor = None
        return cll_exp_tensor, cll_mut_tensor

    def add_drug_data_to_df(self):
        self.read_cmpd_df()
        self.cmpd_df['DrugFeature'] = None
        for i in range(len(self.cmpd_df['cpd_smiles'])):
            drug_feature = []
            rdkit_mol = Chem.MolFromSmiles(self.cmpd_df['cpd_smiles'][i])
            if 'rdkit_des' in self.drug_feat:
                drug_feature += list(Descriptors.CalcMolDescriptors(rdkit_mol).values())
            if 'rdkit_fp' in self.drug_feat:
                drug_feature += list(Chem.RDKFingerprint(rdkit_mol).ToList())
            if 'graph' in self.drug_feat:
                mol_graph = tg.utils.from_smiles(self.cmpd_df['cpd_smiles'][i])
                mol_graph = mol_graph.to('cuda')
                mol_graph.x = torch.tensor(mol_graph.x, dtype=torch.float32)
                mol_graph.edge_attr = torch.tensor(mol_graph.edge_attr, dtype=torch.float32)
                self.cmpd_df['DrugFeature'][i] = mol_graph
            if drug_feature:
                self.cmpd_df['DrugFeature'][i] = drug_feature

    def create_tensor_feat_drug(self):

        self.read_cmpd_df()
        self.add_drug_data_to_df()
        self.read_response_df()
        cmpd_feat_df = self.cmpd_df[['master_cpd_id', 'DrugFeature']]
        cmpd_feat_df = cmpd_feat_df.set_index('master_cpd_id')
        cmpd_feat_df = cmpd_feat_df.reindex(self.response_df['cpdid'])
        if 'graph' in self.drug_feat:
            drug_feat = list(cmpd_feat_df['DrugFeature'])
        else:
            list_feat = []
            for feat in cmpd_feat_df['DrugFeature']:
                list_feat.append(feat)
            drug_feat = torch.tensor(list_feat)
        return drug_feat

    def create_train_test_drug(self):
        if 'graph' in self.drug_feat:
            drug_feat = self.create_tensor_feat_drug()
            train_drug = drug_feat[:-1 * int(len(drug_feat) * self.test_percentage)]
            test_drug = drug_feat[-1 * int(len(drug_feat) * self.test_percentage):]
        else:
            drug_feat = self.create_tensor_feat_drug()
            train_drug = drug_feat[:-1 * int(len(drug_feat) * self.test_percentage)]
            train_drug = train_drug.to(torch.float32)
            train_drug = train_drug.to('cuda')
            # train_drug = torch.nan_to_num(train_drug, nan=0)
            # train_drug = torch.clamp(train_drug, min=0, max=1)
            # train_drug = torch.nn.functional.normalize(train_drug, dim=0)

            test_drug = drug_feat[-1 * int(len(drug_feat) * self.test_percentage):]
            test_drug = test_drug.to(torch.float32)
            test_drug = test_drug.to('cuda')
            # test_drug = torch.nan_to_num(test_drug, nan=0)
            # test_drug = torch.clamp(test_drug, min=0, max=1)
            # test_drug = torch.nn.functional.normalize(test_drug, dim=0)

        return train_drug, test_drug

    def create_train_test_cll(self):
        cll_exp, cll_mut = self.create_tensor_feat_cll()
        if "gene_exp" in self.cll_feat:
            train_exp = cll_exp[:-1 * int(len(cll_exp) * self.test_percentage)]
            train_exp = train_exp.to(torch.float32)
            train_exp = train_exp.to('cuda')
            # train_cll = torch.nan_to_num(train_cll, nan=0)
            # train_cll = torch.clamp(train_cll, min=0, max=1)
            # train_cll = torch.nn.functional.normalize(train_cll, dim=0)

            test_exp = cll_exp[-1 * int(len(cll_exp) * self.test_percentage):]
            test_exp = test_exp.to(torch.float32)
            test_exp = test_exp.to('cuda')
            # test_cll = torch.nan_to_num(test_cll, nan=0)
            # test_cll = torch.clamp(test_cll, min=0, max=1)
            # test_cll = torch.nn.functional.normalize(test_cll, dim=0)
        else:
            train_exp = None
            test_exp = None
        if "gene_mut" in self.cll_feat:
            train_mut = cll_mut[:-1 * int(len(cll_mut) * self.test_percentage)]
            train_mut = train_mut.to(torch.float32)
            train_mut = train_mut.to('cuda')
            # train_cll = torch.nan_to_num(train_cll, nan=0)
            # train_cll = torch.clamp(train_cll, min=0, max=1)
            # train_cll = torch.nn.functional.normalize(train_cll, dim=0)

            test_mut = cll_mut[-1 * int(len(cll_mut) * self.test_percentage):]
            test_mut = test_mut.to(torch.float32)
            test_mut = test_mut.to('cuda')
            # test_cll = torch.nan_to_num(test_cll, nan=0)
            # test_cll = torch.clamp(test_cll, min=0, max=1)
            # test_cll = torch.nn.functional.normalize(test_cll, dim=0)
        else:
            train_mut = None
            test_mut = None
        return train_exp, test_exp, train_mut, test_mut

    def create_train_test_label(self):
        self.read_response_df()
        train_label = self.response_df['auc'][:-1 * int(len(self.response_df) * self.test_percentage)]
        train_label = torch.from_numpy(np.array(train_label))
        train_label = train_label.reshape((-1, 1)).to(torch.float32)
        train_label = train_label.to('cuda')
        test_label = self.response_df['auc'][-1 * int(len(self.response_df) * self.test_percentage):]
        test_label = torch.from_numpy(np.array(test_label))
        test_label = test_label.reshape((-1, 1)).to(torch.float32)
        test_label = test_label.to('cuda')
        return train_label, test_label

    def get_exp_dim(self):
        return self.q

    def get_mut_dim(self):
        return self.q

    def get_drug_dim(self):
        if 'graph' in self.drug_feat:
            drug_dim = 9
        else:
            self.add_drug_data_to_df()
            drug_dim = len(self.cmpd_df['DrugFeature'][0])
        return drug_dim


class CTRPDatasetTorch(Dataset):

    def __init__(self, ctrp_handler, train):
        self.ctrp_handler = ctrp_handler
        self.train = train
        if self.train:
            self.train_exp = self.ctrp_handler.create_train_test_cll()[0]
            self.train_mut = self.ctrp_handler.create_train_test_cll()[2]
            self.train_drug = self.ctrp_handler.create_train_test_drug()[0]
            self.train_label = self.ctrp_handler.create_train_test_label()[0]
        else:
            self.test_exp = self.ctrp_handler.create_train_test_cll()[1]
            self.test_mut = self.ctrp_handler.create_train_test_cll()[3]
            self.test_drug = self.ctrp_handler.create_train_test_drug()[1]
            self.test_label = self.ctrp_handler.create_train_test_label()[1]

    def __len__(self):
        if self.train:
            lenght = len(self.train_label)
        else:
            lenght = len(self.test_label)
        return lenght

    def __getitem__(self, idx):
        if self.train:
            X = self.train_exp[idx], self.train_mut[idx], self.train_drug[idx]
            y = self.train_label[idx]
        else:
            X = self.test_exp[idx], self.test_mut[idx], self.test_drug[idx]
            y = self.test_label[idx]
        return X, y
