import pandas as pd
from rdkit import Chem
import torch
import numpy as np
import json
import torch_geometric as tg
import rdkit
from src.datahandler.netprop import NetProp
from tqdm import tqdm
from torch_geometric.data import DataLoader
from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore


class CTRPHandler:
    def __init__(self):
        """
        Data handling class. It is used after data is wrangled and dimensionality reduction is done
        by network propagation.
        """
        with open('../configs/config.json') as config_file:
            config = json.load(config_file)
        self.top_k_netprop = config['network_propagation']['top_k']
        self.batch_size = config['datahandler']['ctrp_handler']['batch_size']
        self.use_netprop = config['datahandler']['ctrp_handler']['use_netprop']
        self.device = config['main']['device']
        self.normalization_method = config['datahandler']['ctrp_handler']['normalization_method']
        task_type = config['main']['task']
        response_task = config['data_wrangling']['response_task']

        if response_task == 'auc':
            self.response_task = 'area_under_curve'
        elif response_task == 'ic50':
            self.response_task = 'apparent_ec50_umol'

        self.propagated_graph_df = None
        self.read_propagated_graph_df()

        self.response_df = None
        self.read_response_df()
        if task_type == 'ranking':
            self.response_df = self.create_chunks(self.response_df)

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
        self.exp_cll_df = pd.read_csv('../data/wrangled/ccle_exp.csv', index_col=0)
        if self.use_netprop:
            selected_index = list(self.propagated_graph_df['0'])
            self.exp_cll_df = self.exp_cll_df.set_index('nana').iloc[:, selected_index]

    def read_cmpd_df(self):
        """
        Read compound data
        """
        self.cmpd_df = pd.read_csv('../data/wrangled/cmpd.csv', index_col=0)

    def read_response_df(self, reverse=False, normalize=True):
        """
        Read response data. Because AUC and IC50 have negative relationship with the response score, all of them got
        negative and then the maximum of response in terms of absolute number has been added to all data.
        :param normalize:
        :param reverse:
        """
        self.response_df = pd.read_csv('../data/wrangled/ctrp.csv')[:100]
        self.response_df = self.add_weight_column(self.response_df, self.response_task,
                                                  reweight='sqrt_inv', lds=True)
        self.response_df = self.response_df.sample(frac=1, random_state=1234).reset_index(drop=True)
        max_auc = self.response_df[self.response_task].max()
        if reverse:
            self.response_df[self.response_task] = self.response_df[self.response_task].apply(lambda x: (x*-1)+max_auc)
        if normalize:
            if self.normalization_method == 'minmax':
                scaler = MinMaxScaler()
                self.response_df[self.response_task] = scaler.fit_transform(self.response_df[self.response_task])
            elif self.normalization_method == 'zscore':
                self.response_df[self.response_task] = zscore(self.response_df[self.response_task])

    @staticmethod
    def create_chunks(df, chunk_size=5):
        """
        Create chunks of data for list-wise ranking without using external variable

        :param df: response df
        :param chunk_size: list_size
        :return: new response df
        """
        grouped = df.groupby('DepMap_ID')
        new_data = []

        for name, group in grouped:
            num_rows = len(group)
            if num_rows < chunk_size:
                repeats = np.ceil(chunk_size / num_rows).astype(int)
                group = pd.concat([group] * repeats, ignore_index=True)
            group = group.sample(frac=1).reset_index(drop=True)  # Shuffle within each group
            chunks = [group.iloc[i:i + chunk_size] for i in range(0, len(group), chunk_size)]
            new_data.extend(chunks[:len(group) // chunk_size])

        return pd.concat(new_data).reset_index(drop=True)

    @staticmethod
    def add_weight_column(df, label_column, reweight='sqrt_inv', max_target=121, lds=False, lds_kernel='gaussian',
                          lds_ks=5,
                          lds_sigma=2):
        # Function to prepare weights
        def prepare_weights(labels):
            value_dict = {x: 0 for x in range(max_target)}
            for label in labels:
                value_dict[min(max_target - 1, int(label))] += 1
            if reweight == 'sqrt_inv':
                value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
            elif reweight == 'inverse':
                value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}
            num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]
            if not len(num_per_label) or reweight == 'none':
                return None

            if lds:
                lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
                smoothed_value = convolve1d(
                    np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
                num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]

            weights = [np.float32(1 / x) for x in num_per_label]
            scaling = len(weights) / np.sum(weights)
            weights = [scaling * x for x in weights]
            return weights

        # Function to get LDS kernel window
        def get_lds_kernel_window(kernel, ks, sigma):
            assert kernel in ['gaussian', 'triang', 'laplace']
            half_ks = (ks - 1) // 2
            if kernel == 'gaussian':
                base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
                kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(
                    gaussian_filter1d(base_kernel, sigma=sigma))
            elif kernel == 'triang':
                kernel_window = triang(ks)
            else:
                laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
                kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(
                    map(laplace, np.arange(-half_ks, half_ks + 1)))
            return kernel_window

        labels = df[label_column].values
        weights = prepare_weights(labels)
        if weights is not None:
            df['weight'] = weights
        return df

    def read_propagated_graph_df(self):
        """
        Read selected node data after network propagation has been done. The top_k can be entered through config.json
        file. top_k can be 20, 30, 40, 50
        """
        self.propagated_graph_df = pd.read_csv('../data/netprop/top_'+str(self.top_k_netprop)+'_chosen_drug.csv',
                                               index_col=0).astype(int)

    def get_reg_y(self):
        y = self.response_df['area_under_curve'].values
        return y

    def load_y(self, y):
        return DataLoader(y, batch_size=self.batch_size)

    def get_reg_weigth(self):
        y = self.response_df['weight'].values
        return y

    def load_weight(self, weight):
        return DataLoader(weight, batch_size=self.batch_size)

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

    def netprop_dim_reduction(self):
        """
           Propagate PPI network with regard to drugs targets in order to find
           top 20, top 30, top 40 and top 50 genes in the big graph as a way
           to reduce dimension.
           csv files can be found in data folder
        """
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
        drug_chosen.to_csv('../data/netprop/top_20_chosen_drug.csv')
        drug_chosen = pd.Series(selected_index_30)
        drug_chosen.to_csv('../data/netprop/top_30_chosen_drug.csv')
        drug_chosen = pd.Series(selected_index_40)
        drug_chosen.to_csv('../data/netprop/top_40_chosen_drug.csv')
        drug_chosen = pd.Series(selected_index_50)
        drug_chosen.to_csv('../data/netprop/top_50_chosen_drug.csv')

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

    def read_drug_target_df(self):
        """
        Read drug target dataframe
        """
        self.drug_target_df = pd.read_csv('../data/wrangled/drug_target.csv')

    def read_ppi_index_df(self):
        """
        Read ppi dataframe
        """
        self.ppi_index_df = pd.read_csv('../data/wrangled/ppi.csv')
