import pandas as pd
import json


class Wrangler:

    def __init__(self):
        with open('config.json') as config_file:
            config = json.load(config_file)
        self.combined_score = config["data_wrangling"]["min_combined_score"]

    def ctrp_unify_dataframes(self):
        """
        unify three dataframes from CTRP dataset: 'experiment_id', 'master_cpd_id' and one or more response metrics
        from v20.data.curves_post_qc.txt | 'master_cpd_id', 'cpd_name', 'cpd_status',
        'gene_symbol_of_protein_target', 'cpd_smiles' from v20.meta.per_compound.txt | 'experiment_id',
        'master_ccl_id' from v20.meta.per_experiment.txt| 'master_ccl_id', 'ccl_name', 'ccle_primary_site',
        'ccle_primary_hist', 'ccle_hist_subtype_1' from v20.meta.per_cell_line.txt
        Then eliminate rows with None drug target: 83060 from 395263 rows

        :return:
        unified datafream
        """

        response_df = pd.read_csv('data/raw/CTRP/v20.data.curves_post_qc.txt', sep='\t')
        cols = ['experiment_id', 'master_cpd_id', 'area_under_curve', 'apparent_ec50_umol', 'pred_pv_high_conc']
        response_df = response_df[cols]

        cmpd_df = pd.read_csv('data/raw/CTRP/v20.meta.per_compound.txt', sep='\t')
        cmpd_df = cmpd_df[['master_cpd_id', 'cpd_name', 'cpd_status', 'gene_symbol_of_protein_target', 'cpd_smiles']]

        res_cmpd = response_df.merge(cmpd_df, on='master_cpd_id')

        exp_df = pd.read_csv('data/raw/CTRP/v20.meta.per_experiment.txt', sep='\t')
        exp_df = exp_df[['experiment_id', 'master_ccl_id']]

        res_cmpd_exp = res_cmpd.merge(exp_df, on='experiment_id').drop_duplicates()

        cll_df = pd.read_csv('data/raw/CTRP/v20.meta.per_cell_line.txt', sep='\t')
        cll_df = cll_df[['master_ccl_id', 'ccl_name', 'ccle_primary_site', 'ccle_primary_hist', 'ccle_hist_subtype_1']]

        res_cmpd_exp_cll = res_cmpd_exp.merge(cll_df, on='master_ccl_id')
        return res_cmpd_exp_cll.dropna(subset=['gene_symbol_of_protein_target'])

    def add_ccle_to_ctrp(self):
        """
        add DepMap_ID to CTRP data frame created before
        :return
        CTRP DataFrame with depmap id added to
        """
        ctrp_df = self.ctrp_unify_dataframes()

        ccle_info = pd.read_csv('data/raw/CCLE/sample_info.csv')
        depmap_df = ccle_info[['stripped_cell_line_name', 'DepMap_ID']]
        depmap_id = depmap_df.set_index('stripped_cell_line_name').reindex(ctrp_df['ccl_name'])
        ctrp_df['DepMap_ID'] = list(depmap_id['DepMap_ID'])

        return ctrp_df.dropna(subset=['DepMap_ID'])

    def gene_exp_adjustment_by_string(self):
        gene_exp = pd.read_csv('data/raw/CCLE/CCLE_expression.csv')
        string_info = pd.read_csv('data/raw/STRING/9606.protein.info.v12.0.txt', sep='\t')
        gene_list = []
        for i in range(len(list(gene_exp.columns)[1:])):
            gene_list.append(list(gene_exp.columns)[1:][i].split()[0])
        string_info = string_info.set_index('preferred_name')
        string_cll = string_info.reindex(gene_list)
        elim_col = string_cll[string_cll['#string_protein_id'].isnull()].index
        gene_list.insert(0, 'nana')
        gene_exp.columns = gene_list
        gene_exp = gene_exp.drop(list(elim_col), axis=1)
        return gene_exp

    def string_ppi_adjustment_by_ccle(self):
        """

        :return:
        """
        gene_exp = self.gene_exp_adjustment_by_string()
        ppi_df = pd.read_csv('data/raw/STRING/9606.protein.links.v12.0.txt', sep=' ')
        string_info = pd.read_csv('data/raw/STRING/9606.protein.info.v12.0.txt', sep='\t')
        string_info = string_info.set_index('preferred_name')
        string_cll = string_info.reindex(gene_exp.columns[1:])
        ccle_pro = list(string_cll.dropna()['#string_protein_id'])
        ppi_df = ppi_df[ppi_df['protein1'].isin(ccle_pro) & ppi_df['protein2'].isin(ccle_pro)]
        ppi_df = ppi_df[ppi_df['combined_score'] > self.combined_score]
        value_index_mapping = {value: index for index, value in enumerate(ccle_pro)}
        ppi_df['protein1'] = ppi_df['protein1'].replace(value_index_mapping)
        ppi_df['protein2'] = ppi_df['protein2'].replace(value_index_mapping)
        return ppi_df

    def add_target_index_to_ctrp(self):

        ctrp_df = self.add_ccle_to_ctrp()
        gene_exp = self.gene_exp_adjustment_by_string()
        ctrp_df['index_target'] = ctrp_df['gene_symbol_of_protein_target'].copy()
        ctrp_df = ctrp_df.assign(index_target=ctrp_df['index_target'].str.split(';')).explode('index_target')
        ctrp_df = ctrp_df.reset_index(drop=True)
        value_index_mapping = {value: index for index, value in enumerate(list(gene_exp.columns[1:]))}
        ctrp_df['index_target'] = ctrp_df['index_target'].replace(value_index_mapping)
        ctrp_df['index_target'] = pd.to_numeric(ctrp_df['index_target'], errors='coerce')
        ctrp_df = ctrp_df.dropna(subset=['index_target']).astype({'index_target': 'int'})
        return ctrp_df

    def save_wrangled_data(self):

        gene_exp = self.gene_exp_adjustment_by_string()
        gene_exp.to_csv('data/wrangled/ccle_exp.csv')

        drug_target = self.add_target_index_to_ctrp()
        drug_target.to_csv('data/wrangled/drug_target.csv')

        cpd_id = list(drug_target['master_cpd_id'].unique())
        ctrp_df = self.add_ccle_to_ctrp()
        ctrp_df = ctrp_df[ctrp_df['master_cpd_id'].isin(cpd_id)]
        ctrp_df = ctrp_df[ctrp_df['DepMap_ID'].isin(list(gene_exp['nana']))]
        ctrp_df.to_csv('data/wrangled/ctrp.csv')
        cmpd_df = pd.DataFrame({'master_cpd_id': list(ctrp_df['master_cpd_id'].unique()),
                                'cpd_smiles': list(ctrp_df['cpd_smiles'].unique())})
        cmpd_df.to_csv('data/wrangled/cmpd.csv')
        ppi = self.string_ppi_adjustment_by_ccle()
        ppi.to_csv('data/wrangled/ppi.csv')
