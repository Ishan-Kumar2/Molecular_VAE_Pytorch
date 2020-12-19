import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Data Prep')
parser.add_argument('file_loc', type=str, help='Path to the downloading dataset and name')
args = parser.parse_args()

chembl22 = pd.read_table(args.file_loc)
print(chembl22['canonical_smiles'][0:10])

df = pd.DataFrame({'SMILES':chembl22['canonical_smiles']})
df.to_csv('./smiles_chembl.csv')