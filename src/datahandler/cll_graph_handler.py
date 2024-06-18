import pandas as pd
from tqdm import tqdm
import numpy as np
from torch_geometric.data import Data
import torch

class CllGraphHandler:
    def __init__(self):
        self.read_csv_files()

    def read_csv_files(self):
        self.exp_df = pd.read_csv('../data/raw/CCLE/CCLE_expression.csv').iloc[:100, :100]
        self.exp_df.columns = [col.split(' ')[0] for col in self.exp_df.columns]
        self.mut_df = pd.read_csv('../data/raw/CCLE/CCLE_mutations.csv')
        self.target_df = pd.read_csv('../data/raw/CTRP/v20.meta.per_compound.txt', sep='\t')
        self.edge_df = pd.read_csv('../data/raw/STRING/9606.protein.links.v12.0.txt', sep=' ')
        self.edge_df = self.edge_df[self.edge_df['combined_score'] >= 400]
        self.string_meta = pd.read_csv('../data/raw/STRING/9606.protein.info.v12.0.txt', sep='\t')
        self.ccle_meta = pd.read_csv('../data/raw/CCLE/sample_info.csv')

    def get_drugs_target(self):
        drug_targets = self.target_df['gene_symbol_of_protein_target'].dropna().str.split(';').explode().unique()
        return drug_targets

    def gene_features(self):
        # Set the first column as the index
        self.exp_df.set_index(self.exp_df.columns[0], inplace=True)
        mut_clls = list(self.mut_df['DepMap_ID'].unique())
        self.exp_df = self.exp_df.reindex(mut_clls).dropna()

        # Precompute drug target information
        drug_targets = self.get_drugs_target()
        is_target = self.exp_df.columns.isin(drug_targets).astype(int)
        # Initialize feature_df with appropriate structure
        feature_df = pd.DataFrame(index=self.exp_df.index, columns=self.exp_df.columns, dtype=object)
        one_hot_vt = pd.get_dummies(self.mut_df['Variant_Type'])
        one_hot_va = pd.get_dummies(self.mut_df['Variant_annotation'])
        self.mut_df = pd.concat([self.mut_df, one_hot_vt], axis=1)
        self.mut_df = pd.concat([self.mut_df, one_hot_va], axis=1)
        columns_to_convert = ['DEL', 'DNP', 'INS', 'SNP', 'TNP', 'isDeleterious',
                              'damaging', 'other conserving', 'other non-conserving', 'silent']
        self.mut_df[columns_to_convert] = self.mut_df[columns_to_convert].astype(float)
        # Create a dictionary for quick mutation lookup
        mutation_dict = self.mut_df.groupby(['DepMap_ID', 'Hugo_Symbol']).apply(
            lambda x: x[['DEL', 'DNP', 'INS', 'SNP', 'TNP', 'isDeleterious',
                         'damaging', 'other conserving', 'other non-conserving',
                         'silent']].values[0] if len(x) > 0 else [0]*len(columns_to_convert)
        ).to_dict()
        # Vectorized approach to create feature vectors
        for gene in tqdm(self.exp_df.columns, desc="Genes"):
            expression_values = self.exp_df[gene]
            gene_is_target = is_target[self.exp_df.columns.get_loc(gene)]
            feature_df[gene] = [
                [expression_values[cell_line]] + list(map(str, mutation_dict.get((cell_line, gene),
                                                                                 [0]*len(columns_to_convert)))) + [str(gene_is_target)]
                for cell_line in self.exp_df.index
            ]
        return feature_df

    def get_edges(self):
        cll_string_index = self.string_meta.set_index('preferred_name').reindex(self.exp_df.columns[1:])
        elim_genes = list(cll_string_index[cll_string_index.isnull().any(axis=1)].index)
        self.exp_df = self.exp_df.drop(columns=elim_genes)
        cll_string_index.dropna(inplace=True)
        string_genes = cll_string_index['#string_protein_id'].reset_index()
        self.edge_df = self.edge_df[
            self.edge_df['protein1'].isin(string_genes['#string_protein_id']) &
            self.edge_df['protein2'].isin(string_genes['#string_protein_id'])
        ]
        replace_dict = string_genes.reset_index().set_index('#string_protein_id')['level_0'].to_dict()
        '''edge_index = self.edge_df.replace(replace_dict)
        print(edge_index)'''
        edge_values = self.edge_df.values
        vectorized_replace = np.vectorize(replace_dict.get)

        # Apply the replacement to both columns
        edge_values[:, 0] = vectorized_replace(edge_values[:, 0])
        edge_values[:, 1] = vectorized_replace(edge_values[:, 1])

        # Convert the numpy array back to a DataFrame
        edge_index = pd.DataFrame(edge_values, columns=self.edge_df.columns)
        edges = edge_index[['protein1', 'protein2']].values.reshape(2, -1)
        edge_attr = edge_index['combined_score'].values.reshape(-1, 1)
        return edges, edge_attr

    def get_graph(self):
        device = 'cuda'
        edge_index, edge_attr = self.get_edges()
        feature_df = self.gene_features()
        graph_list = []
        for i in range(len(feature_df)):
            x = feature_df.iloc[i, :].tolist()
            x = np.vstack(x).astype(np.float32)
            edge_index = edge_index.astype(np.int32)
            edge_attr = edge_attr.astype(np.float32)
            cll_graph = Data(x=torch.tensor(x, dtype=torch.float).to(device),
                             edge_index=torch.tensor(edge_index, dtype=torch.int32).to(device),
                             edge_attr=torch.tensor(edge_attr, dtype=torch.float)).to(device)
            graph_list.append(cll_graph)
        print(feature_df.index)
        return graph_list
