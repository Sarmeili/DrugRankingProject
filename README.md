# Precision Drug Response Prediction in Cancer Cell Lines: Analyzing Omics, Protein-Protein Interactions, and Drug-Target Interactions using Deep Learning and Graph Neural Networks

This project aims to predict the response of drugs on cancer cell lines using drug data, gene expression data, and various interaction datasets. Please note that this project is still in progress, and there may be some instabilities when running the code. The results of my experiments will be shared soon.

## Project Configuration

### Normalization Method
Specify the normalization method to be used:
- `zscore`
- `minmax`

### Task
Define the task type:
- `ranking`
- `regression`

### Response Task
Choose the response task:
- `auc`
- `ic50`

## Data Files

Download and place the following data files in the `data/wrangled/` directory. Ensure to rename them appropriately after downloading:

1. **Processed CTRP Data**

```shell
wget https://figshare.com/ndownloader/files/43719579 -O data/wrangled/ctrp.csv
```

2. **CCLE Expression Data**

```shell
wget https://figshare.com/ndownloader/files/43719501 -O data/wrangled/ccle_exp.csv
```

3. **Drug Target Data**
```shell
wget https://figshare.com/ndownloader/files/43719537 -O data/wrangled/drug_target.csv
```

4. **Raw CCLE Mutation Data**
```shell
wget https://ndownloader.figshare.com/files/34989940 -O data/raw/CCLE_mutations.csv
```

5. **Raw CCLE Expression Data**
```shell
wget https://ndownloader.figshare.com/files/34989919 -O data/raw/CCLE/CCLE_expression.csv
```

6. **Raw CTRP Data**
```shell
mkdir -p /content/DrugRankingProject/data/raw/CTRP
wget https://ctd2-data.nci.nih.gov/Public/Broad/CTRPv2.0_2015_ctd2_ExpandedDataset/CTRPv2.0_2015_ctd2_ExpandedDataset.zip -O data/raw/CTRP/CTRPv2.0_2015_ctd2_ExpandedDataset.zip
unzip data/raw/CTRP/CTRPv2.0_2015_ctd2_ExpandedDataset.zip -d /content/DrugRankingProject/data/raw/CTRP
```

7. **Raw STRING Data**
```shell
mkdir -p /content/DrugRankingProject/data/raw/STRING
wget https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz -O data/raw/STRING/9606.protein.links.v12.0.txt.gz
wget https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz -O data/raw/STRING/9606.protein.info.v12.0.txt.gz
gunzip data/raw/STRING/9606.protein.links.v12.0.txt.gz -d /content/DrugRankingProject/data/raw/STRING
gunzip data/raw/STRING/9606.protein.info.v12.0.txt.gz -d /content/DrugRankingProject/data/raw/STRING
```

## Installation
1. **Clone the repository**
```shell
git clone https://github.com/Sarmeili/DrugRankingProject.git
cd DrugRankingProject
```
2. **Install the requirements**

Make sure you have Python installed. It's recommended to use a virtual environment.
```shell
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```
Install the required dependencies by running:
```shell
pip install -r requirements.txt
```