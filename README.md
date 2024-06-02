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