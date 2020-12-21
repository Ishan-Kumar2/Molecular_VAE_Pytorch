import regex as re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import functools
#from rdkit.Chem.Descriptors import MolWt
#import imblearn
#from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

from matplotlib import colors
#from rdkit.Chem import Draw
#from rdkit.Chem.Draw import MolToImage
#import rdkit
#import rdkit.Chem as Chem
from PIL import Image  
import PIL 
import re
# check version number
#import imblearn
#from imblearn.over_sampling import RandomOverSampler
#oversample = RandomOverSampler(sampling_strategy='minority')


SMILES_COL_NAME = 'SMILES'


SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""
regex = re.compile(SMI_REGEX_PATTERN)

def tokenizer(smiles_string):
	tokens = [token for token in regex.findall(smiles_string)]
	return tokens

def atomwise_tokenizer(smi, exclusive_tokens = None):
    """
    Tokenize a SMILES molecule at atom-level:
        (1) 'Br' and 'Cl' are two-character tokens
        (2) Symbols with bracket are considered as tokens
    exclusive_tokens: A list of specifical symbols with bracket you want to keep. e.g., ['[C@@H]', '[nH]'].
    Other symbols with bracket will be replaced by '[UNK]'. default is `None`.
    """
    
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]

    if exclusive_tokens:
        for i, tok in enumerate(tokens):
            if tok.startswith('['):
                if tok not in exclusive_tokens:
                    tokens[i] = '[UNK]'
    return tokens

def build_vocab(data):
	vocab_ = set()
	smiles = list(data[SMILES_COL_NAME])

	for ex in smiles:
		for letter in tokenizer(ex):
			vocab_.add(letter)
	
	vocab={}
	vocab['<PAD>'] = 0
	vocab['<UNK>'] = 1
	for i,letter in enumerate(vocab_):
		vocab[letter]=i+2
	inv_dict= {num: char for char, num in vocab.items()}
	inv_dict[0] = ''
	return vocab, inv_dict

def custom_vocab():
	vocab_ = {'[c-]', '[SeH]', '[N]', '[C@@]', '[Te]', '[OH+]', 'n', '[AsH]', '[B]', 'b', '[S@@]', 'o', ')', '[NH+]', '[SH]', 'O', 'I', '[C@]', '-', '[As+]', '[Cl+2]', '[P+]', '[o+]', '[C]', '[C@H]', '[CH2]', '\\', 'P', '[O-]', '[NH-]', '[S@@+]', '[te]', '[s+]', 's', '[B-]', 'B', 'F', '=', '[te+]', '[H]', '[C@@H]', '[Na]', '[Si]', '[CH2-]', '[S@+]', 'C', '[se+]', '[cH-]', '6', 'N', '[IH2]', '[As]', '[Si@]', '[BH3-]', '[Se]', 'Br', '[C+]', '[I+3]', '[b-]', '[P@+]', '[SH2]', '[I+2]', '%11', '[Ag-3]', '[O]', '9', 'c', '[N-]', '[BH-]', '4', '[N@+]', '[SiH]', '[Cl+3]', '#', '(', '[O+]', '[S-]', '[Br+2]', '[nH]', '[N+]', '[n-]', '3', '[Se+]', '[P@@]', '[Zn]', '2', '[NH2+]', '%10', '[SiH2]', '[nH+]', '[Si@@]', '[P@@+]', '/', '1', '[c+]', '[S@]', '[S+]', '[SH+]', '[B@@-]', '8', '[B@-]', '[C-]', '7', '[P@]', '[se]', 'S', '[n+]', '[PH]', '[I+]', '5', 'p', '[BH2-]', '[N@@+]', '[CH]', 'Cl'}
	vocab={}
	vocab['<PAD>'] = 0
	vocab['<UNK>'] = 1
	for i,letter in enumerate(vocab_):
		vocab[letter]=i+1
	inv_dict= {num: char for char, num in vocab.items()}
	inv_dict[0] = ''
	return vocab, inv_dict

def make_one_hot(data,vocab,max_len=120):
	data_one_hot=np.zeros((len(data),max_len,len(vocab)))
	for i, smiles in enumerate(data):
		
		smiles = tokenizer(smiles)
		smiles = smiles[:120] +['<PAD>']*(max_len-len(smiles))

		for j,letter in enumerate(smiles):
			if letter in vocab.keys():
				data_one_hot[i,j,vocab[letter]] = 1
			else:
				data_one_hot[i,j,vocab['<UNK>']] = 1
	return data_one_hot


def oversample(input,labels):
	oversample = RandomOverSampler(sampling_strategy='minority')
	X_oversampled,y_oversampled = oversample.fit_resample(input,labels)
	return X_oversampled,y_oversampled

def get_ratio_classes(labels):
	print('Number of 1s in dataset -- {} Percentage -- {:.3f}%'.format(labels[labels==1].shape[0],
															 labels[labels==1].shape[0]/len(labels)))

	print('Number of 0s in dataset -- {} Percentage -- {:.3f}%'.format(labels[labels==0].shape[0],
															 labels[labels==0].shape[0]/len(labels)))


def split_data(input,output,test_size=0.20):
	X_train, X_test, y_train, y_test = train_test_split(input, output, 
												  test_size=test_size, 
												  stratify=output,
												  random_state=42)

	return X_train, X_test, y_train, y_test


def get_image(mol, atomset, name):    
	hcolor = colors.to_rgb('green')
	if atomset is not None:
		#highlight the atoms set while drawing the whole molecule.
		img = MolToImage(mol, size=(600, 600),fitImage=True, highlightAtoms=atomset,highlightColor=hcolor)
	else:
		img = MolToImage(mol, size=(400, 400),fitImage=True)

	
	img = img.save(name + ".jpg") 
	return img

def onehot_to_smiles(onehot, inv_vocab):
	#print(np.where(onehot == 1))
	#return "".join(map(lambda x: inv_vocab[x], np.where(onehot == 1)[1]))
	#print(onehot,onehot.shape)
	#print(onehot.argmax(axis=2)[0])
	return "".join(inv_vocab[let.item()] for let in onehot.argmax(axis=2)[0])
	


def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol)
    return mol


def add_img(onehot, inv_vocab, name):
	smiles = onehot_to_smiles(onehot, inv_vocab)
	mol = get_mol(smiles)
	get_image(mol, {}, name)

import h5py
def load_dataset(filename, split = True):
    #h5f = h5py.File(filename, 'r')
    data = pd.read_hdf(filename, 'table')
    print(h5f)
    if split:
        data_train = h5f['data_train'][:]
    else:
        data_train = None
    data_test = h5f['data_test'][:]
    charset =  h5f['charset'][:]
    h5f.close()
    if split:
        return (data_train, data_test, charset)
    else:
        return (data_test, charset)



if __name__ == '__main__':
	#a,b,c=load_dataset('/home/ishan/Desktop/Chem_AI/keras-molecules-master/data/smiles_50k.h5')

	dat = pd.read_csv('./data/smiles_chembl.csv')
	dat = dat.tail(50000)
	print(dat.head(10))
	



	vocab,inv_dict = build_vocab(dat)
	print("Vocab",vocab)
	print("Vocab Size",len(vocab))

	vocab_2, inv_dict_2 = custom_vocab()
	print("Len of Custom Vocab",len(vocab_2))
	data_one_hot = make_one_hot(dat[SMILES_COL_NAME],vocab)
	print(data_one_hot.shape)
	####Checking onehot_to_smiles
	print("Original",data[SMILES_COL_NAME][5])
	print("Recon",onehot_to_smiles(data_one_hot[5], inv_dict))
	print(data[SMILES_COL_NAME][5] == onehot_to_smiles(data_one_hot[5], inv_dict) )
	#####

	add_img(data_one_hot[5], inv_dict, 'checking')

	print("One Hot Train Data Shape",data_one_hot.shape)
	print(data_one_hot[0])
