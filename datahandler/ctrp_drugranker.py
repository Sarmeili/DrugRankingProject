import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import torch


class CTRPHandler:

    def __init__(self, source, data_volume):
        self.source = source
        self.data_volume = data_volume

    def read_data_as_df(self):
        if self.source == 'DrugRanker':
            self.cmpd_df = pd.read_csv('data/ctrp_drugranker/cmpd_id_name_group_smiles.txt', sep='\t')
            self.reponse_df = pd.read_csv('data/ctrp_drugranker/final_list_auc.txt')
            self.reponse_df = self.reponse_df[:int(len(self.reponse_df) * self.data_volume)]
            self.exp_ccl_df = pd.read_csv('data/CCLE_expression.csv', index_col=0)

    def create_tensor_feat_cll(self):

        self.read_data_as_df()
        feat_ccl = self.exp_ccl_df.reindex(self.reponse_df['broadid'])
        ccl_feat_tensor = torch.from_numpy(feat_ccl.to_numpy())
        return ccl_feat_tensor

    def add_drug_data_to_df(self, features_type):
        self.read_data_as_df()
        self.cmpd_df['DrugFeature'] = None
        if self.source == 'DrugRanker':
            for i in range(len(self.cmpd_df['cpd_smiles'])):
                drug_feature = []
                rdkit_mol = Chem.MolFromSmiles(self.cmpd_df['cpd_smiles'][i])
                if 'rdkit_des' in features_type:
                    drug_feature += list(Descriptors.CalcMolDescriptors(rdkit_mol).values())
                if 'rdkit_fp' in features_type:
                    drug_feature += list(Chem.RDKFingerprint(rdkit_mol).ToList())
                self.cmpd_df['DrugFeature'][i] = drug_feature

    def create_tensor_feat_drug(self):

        self.read_data_as_df()
        self.add_drug_data_to_df(['rdkit_des', 'rdkit_fp'])
        cmpd_feat_df = self.cmpd_df[['master_cpd_id', 'DrugFeature']]
        cmpd_feat_df = cmpd_feat_df.set_index('master_cpd_id')
        cmpd_feat_df = cmpd_feat_df.reindex(self.reponse_df['cpdid'])
        list_feat = []
        for feat in cmpd_feat_df['DrugFeature']:
            list_feat.append(feat)
        drug_feat_tensor = torch.tensor(list_feat)
        return drug_feat_tensor
