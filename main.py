from model import *
from data_preprocessing import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim


data = pd.read_csv('/home/ishan/Desktop/Chem_AI/MolPMoFiT/data/QSAR/bbbp.csv')

smiles = data['smiles']
labels = np.array(data['p_np'])
print("Example Smiles",smiles[0:10])
vocab,inv_dict = build_vocab(data)
vocab_size = len(vocab)
data = make_one_hot(data['smiles'],vocab)
print("One Hot Data Shape",data.shape)


print("Before Over Sampling")
get_ratio_classes(labels)

oversample = False
if oversample:
	print(data.shape,labels.shape)
	data,labels = oversample(data,labels)
	print("After Over Sampling")
	get_ratio_classes(labels_oversampled)

X_train, X_test, y_train, y_test = split_data(data,labels)
print("Train Data Shape--{} Labels Shape--{} ".format(X_train.shape,y_train.shape))
print("Test Data Shape--{} Labels Shape--{} ".format(X_test.shape,y_test.shape))





input_dim = 120 * 71
hidden_dim = 200
hidden_2 = 120
latent = 60

use_vae = False
if use_vae:
	enc = Encoder(input_dim,hidden_dim,hidden_2)
	dec = Decoder(input_dim,hidden_dim,latent)
	model = VAE(enc,dec,latent)
	model.get_num_params()
else:
	enc = Conv_Encoder()
	dec = GRU_Decoder(vocab_size)
	model = Molecule_VAE(enc, dec)
	model.get_num_params()
	
criterion = nn.MSELoss()
#bce_loss = F.binary_cross_entropy()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

batch_size = 8
n_epochs = 10

dataloader = torch.utils.data.DataLoader(X_train, batch_size=batch_size,
                                             shuffle=False, num_workers=2,drop_last = True)

print("###########################################################################")
for epoch in range(n_epochs):
	epoch_loss = 0

	for i, data in enumerate(dataloader, 0):
		
		inputs = data.float()
		#inputs = inputs.reshape(batch_size, -1).float()
		optimizer.zero_grad()

		input_recon = model(inputs)
		
		latent_loss_val = latent_loss(model.z_mean, model.z_sigma)
		loss = F.binary_cross_entropy(input_recon, inputs, size_average=False) + latent_loss_val
		loss.backward()
		optimizer.step()
		epoch_loss += loss.item()

	print("Epoch -- {} Loss -- {:.3f}".format(epoch,epoch_loss/len(data)))
	add_img(inputs[0], inv_dict, "Original_"+str(epoch))
	add_img(model(inputs[0:1]), inv_dict, "Recon_"+str(epoch))




