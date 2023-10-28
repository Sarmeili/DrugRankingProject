import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import torch
import numpy as np
from torch.utils.data import Dataset
import json
from scipy.stats import zscore


class CTRPHandler:

    def __init__(self, source, data_volume):
        self.mean = None
        self.std = None
        with open('config.json') as config_file:
            config = json.load(config_file)
        self.drug_feat = config['datahandler']['ctrp_drugranker']['drug_feat']
        self.is_pca = config['datahandler']['ctrp_drugranker']['cll_is_pca']
        self.test_percentage = config['datahandler']['ctrp_drugranker']['test_percentage']
        self.response_df = None
        self.exp_ccl_df = None
        self.cmpd_df = None
        self.source = source
        self.data_volume = data_volume

    def z_score_calculation(self, input):
        return (input - self.mean)/self.std

    def read_data_as_df(self):
        if self.source == 'DrugRanker':
            self.cmpd_df = pd.read_csv('data/ctrp_drugranker/cmpd_id_name_group_smiles.txt', sep='\t')
            self.response_df = pd.read_csv('data/ctrp_drugranker/final_list_auc.txt')
            self.mean = self.response_df['auc'].mean()
            self.std = self.response_df['auc'].std()
            self.response_df['auc'] = self.response_df['auc'].apply(self.z_score_calculation)
            self.response_df = self.response_df[int(len(self.response_df) * self.data_volume[0]):int(
                len(self.response_df) * self.data_volume[1])]
            self.exp_ccl_df = pd.read_csv('data/CCLE_expression.csv', index_col=0).apply(zscore)

    def create_tensor_feat_cll(self):

        self.read_data_as_df()
        exp_cll_df = self.exp_ccl_df.astype('float32')
        feat_ccl = exp_cll_df.reindex(self.response_df['broadid'])
        ccl_feat_tensor = torch.from_numpy(feat_ccl.to_numpy())
        if self.is_pca:
            pca = torch.pca_lowrank(ccl_feat_tensor, q=3000)
            ccl_feat_tensor = pca[0]
        return ccl_feat_tensor

    def add_drug_data_to_df(self):
        self.read_data_as_df()
        self.cmpd_df['DrugFeature'] = None
        if self.source == 'DrugRanker':
            for i in range(len(self.cmpd_df['cpd_smiles'])):
                drug_feature = []
                rdkit_mol = Chem.MolFromSmiles(self.cmpd_df['cpd_smiles'][i])
                if 'rdkit_des' in self.drug_feat:
                    drug_feature += list(Descriptors.CalcMolDescriptors(rdkit_mol).values())
                if 'rdkit_fp' in self.drug_feat:
                    drug_feature += list(Chem.RDKFingerprint(rdkit_mol).ToList())
                self.cmpd_df['DrugFeature'][i] = drug_feature

    def create_tensor_feat_drug(self):

        self.read_data_as_df()
        self.add_drug_data_to_df()
        cmpd_feat_df = self.cmpd_df[['master_cpd_id', 'DrugFeature']]
        cmpd_feat_df = cmpd_feat_df.set_index('master_cpd_id')
        cmpd_feat_df = cmpd_feat_df.reindex(self.response_df['cpdid'])
        list_feat = []
        for feat in cmpd_feat_df['DrugFeature']:
            list_feat.append(feat)
        drug_feat_tensor = torch.tensor(list_feat)
        return drug_feat_tensor

    def create_train_test_drug(self):
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
        cll_feat = self.create_tensor_feat_cll()
        train_cll = cll_feat[:-1 * int(len(cll_feat) * self.test_percentage)]
        train_cll = train_cll.to(torch.float32)
        train_cll = train_cll.to('cuda')
        train_cll = torch.nan_to_num(train_cll, nan=0)
        # train_cll = torch.clamp(train_cll, min=0, max=1)
        # train_cll = torch.nn.functional.normalize(train_cll, dim=0)

        test_cll = cll_feat[-1 * int(len(cll_feat) * self.test_percentage):]
        test_cll = test_cll.to(torch.float32)
        test_cll = test_cll.to('cuda')
        test_cll = torch.nan_to_num(test_cll, nan=0)
        # test_cll = torch.clamp(test_cll, min=0, max=1)
        # test_cll = torch.nn.functional.normalize(test_cll, dim=0)

        return train_cll, test_cll

    def create_train_test_label(self):
        self.read_data_as_df()
        train_label = self.response_df['auc'][:-1 * int(len(self.response_df) * self.test_percentage)]
        train_label = torch.from_numpy(np.array(train_label))
        train_label = train_label.reshape((-1, 1)).to(torch.float32)
        train_label = train_label.to('cuda')
        test_label = self.response_df['auc'][-1 * int(len(self.response_df) * self.test_percentage):]
        test_label = torch.from_numpy(np.array(test_label))
        test_label = test_label.reshape((-1, 1)).to(torch.float32)
        test_label = test_label.to('cuda')
        return train_label, test_label


class CTRPDatasetTorch(Dataset):

    def __init__(self, ctrp_handler, train):
        self.ctrp_handler = ctrp_handler
        self.train = train
        if self.train:
            self.train_cll = self.ctrp_handler.create_train_test_cll()[0]
            self.train_drug = self.ctrp_handler.create_train_test_drug()[0]
            self.train_label = self.ctrp_handler.create_train_test_label()[0]
        else:
            self.test_cll = self.ctrp_handler.create_train_test_cll()[1]
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
            X = self.train_cll[idx], self.train_drug[idx]
            y = self.train_label[idx]
        else:
            X = self.test_cll[idx], self.test_drug[idx]
            y = self.test_label[idx]
        return X, y
