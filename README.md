# Precision Drug Ranking in Cancer Cell Lines: A Comprehensive Analysis of Omics, Protein-Protein Interaction, and Drug-Target Interaction using Deep Learning and Graph Neural Networks

This project aims to enhance precision drug ranking in cancer cell lines by leveraging deep learning and graph neural networks to analyze omics data, protein-protein interactions, and drug-target interactions.

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

1. **CTRP Data**

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

## Installation

Install the required dependencies by running:
```shell
pip install -r requirements.txt
```