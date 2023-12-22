22Q1 CCLE Exp at `data/CCLE_expression.csv`

22Q1 CCLE Mut at `data/CCLE_mutations_bool_damaging.csv`

drug_feat : list of "rdkit_fp", "rdkit_des", "graph"
cll_feat : list of "gene_exp", "gene_mut"
task : one of "ranking", "regression"
response_type : list of "auc", "ic50", "senscore"

ctrp.csv : wget https://figshare.com/ndownloader/files/43719579 -P data/wrangled/
ccle_exp.csv : wget https://figshare.com/ndownloader/files/43719501 -P data/wrangled/
drug_target.csv : wget https://figshare.com/ndownloader/files/43719537 -P data/wrangled/
