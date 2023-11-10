import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import torch
import numpy as np
from torch.utils.data import Dataset
import json
from scipy.stats import zscore
import torch_geometric as tg


class CTRPHandler:

    def __init__(self, data_volume):
        self.data_volume = data_volume
        self.mean = None
        self.std = None
        self.cmpd_df = None
        self.exp_cll_df = None
        self.mut_cll_df = None
        self.response_df = None
        with open('config.json') as config_file:
            config = json.load(config_file)
        self.drug_feat = config['datahandler']['ctrp_drugranker']['drug_feat']
        self.cll_feat = config['datahandler']['ctrp_drugranker']['cll_feat']
        self.is_pca = config['datahandler']['ctrp_drugranker']['dim_reduction']['pca']['is_pca']
        self.q = config['datahandler']['ctrp_drugranker']['dim_reduction']['pca']['q']
        self.test_percentage = config['datahandler']['ctrp_drugranker']['test_percentage']
        self.source = config['datahandler']['ctrp_drugranker']['source']
        self.last_layer = config['model_experiments']['graphmol_mlp']['last_layer']

    def z_score_calculation(self, x):
        return (x - self.mean) / self.std

    def read_cll_df(self):
        if "gene_exp" in self.cll_feat:
            self.exp_cll_df = pd.read_csv('data/CCLE_expression.csv', index_col=0)
        if "gene_mut" in self.cll_feat:
            self.mut_cll_df = pd.read_csv('data/CCLE_mutations_bool_damaging.csv', index_col=0)

    def read_cmpd_df(self):
        if self.source == 'DrugRanker':
            self.cmpd_df = pd.read_csv('data/ctrp_drugranker/cmpd_id_name_group_smiles.txt', sep='\t')

    def read_response_df(self):
        if self.source == 'DrugRanker':
            self.response_df = pd.read_csv('data/ctrp_drugranker/final_list_auc.txt')
            # self.mean = self.response_df['auc'].mean()
            # self.std = self.response_df['auc'].std()
            # self.response_df['auc'] = self.response_df['auc'].apply(self.z_score_calculation)
            self.response_df = self.response_df[int(len(self.response_df) * self.data_volume[0]):int(
                len(self.response_df) * self.data_volume[1])]

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
                self.cmpd_df['DrugFeature'][i] = tg.utils.from_smiles(self.cmpd_df['cpd_smiles'][i]).to('cuda:0')
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
