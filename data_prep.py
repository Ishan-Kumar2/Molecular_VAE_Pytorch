import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Data Prep')
parser.add_argument('file_loc', type=str, help='Path to the downloading dataset and name')
parser.add_argument('smiles_col_name', type=str, help='Name of the Column of SMILES')
parser.add_argument('save_path', type=str, help='Save Path of the Processed Data (CSV)')
parser.add_argument('len',type=int,help='Number of Training Points needed in Train and Val Data')
args = parser.parse_args()

chembl22 = pd.read_table(args.file_loc)
print(chembl22[args.smiles_col_name][0:10])

df = pd.DataFrame({'SMILES':chembl22[args.smiles_col_name]})
df.to_csv(args.save_path+'.csv')

df = pd.DataFrame({'SMILES':chembl22[args.smiles_col_name]})
df.to_csv(args.save_path+'.csv')

#Shuffling the Data to remove any sampling bias
df.sample(frac=1)

train_data = df.head(args.len)
val_data = df.tail(args.len)

train_data.to_csv(args.save_path+'_train.csv')
val_data.to_csv(args.save_path+'_val.csv')
