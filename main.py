from model import *
from data_preprocessing import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import argparse
import random


NUM_EPOCHS = 1
BATCH_SIZE = 60
LATENT_DIM = 292
RANDOM_SEED = 42
LR = 0.0001
DYN_LR = True

def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular VAE network')
    parser.add_argument('data', type=str, help='Path to the dataset and name')
    parser.add_argument('save_loc', type=str,
                        help='Where to save the trained model. If this file exists, it will be opened and resumed.')
    parser.add_argument('--epochs', type=int, metavar='N', default=NUM_EPOCHS,
                        help='Number of epochs to run during training.')
    parser.add_argument('--model_type', type=str, help='Can Train either Molecular VAE Arch or Vanilla FC VAE',
    						default='mol_vae')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT_DIM,
                        help='Dimensionality of the latent variable.')
    parser.add_argument('--batch_size', type=int, metavar='N', default=BATCH_SIZE,
                        help='Number of samples to process per minibatch during training.')
    parser.add_argument('--lr', type=float, metavar='N', default=LR,
                        help='Learning Rate for training.')

    
    return parser.parse_args()

def main():
	args = get_arguments()

	data = pd.read_csv(args.data)

	smiles = data['smiles']
	#labels = np.array(data['p_np'])
	labels = np.zeros((len(smiles),1))
	print("Example Smiles",smiles[0:10])

	##Building the Vocab from DeepChem's Regex
	vocab, inv_dict = build_vocab(data)
	vocab_size = len(vocab)
	
	##Converting to One Hot Vectors
	data = make_one_hot(data['smiles'],vocab)
	print("Input Data Shape",data.shape)
	
	##Checking ratio for Classification Datasets
	#print("Ratio of Classes")
	#get_ratio_classes(labels)

	##To oversample datasets if dataset is imbalanced
	oversample = False
	if oversample:
		print(data.shape,labels.shape)
		data,labels = oversample(data,labels)
		print("After Over Sampling")
		get_ratio_classes(labels_oversampled)

	##Train Test Split
	X_train, X_test, y_train, y_test = split_data(data,labels)
	print("Train Data Shape--{} Labels Shape--{} ".format(X_train.shape,y_train.shape))
	print("Test Data Shape--{} Labels Shape--{} ".format(X_test.shape,y_test.shape))


	use_vae = args.model_type == 'mol_vae'
	if use_vae:
		##Using Molecular VAE Arch as in the Paper with Conv Encoder and GRU Decoder
		enc = Conv_Encoder()
		dec = GRU_Decoder(vocab_size)
		model = Molecule_VAE(enc, dec)
		model.get_num_params()
	else:
		#Using FC layers for both Encoder and Decoder
		input_dim = 120 * 71
		hidden_dim = 200
		hidden_2 = 120
		latent = 60
		enc = Encoder(input_dim,hidden_dim,hidden_2)
		dec = Decoder(input_dim,hidden_dim,latent)
		model = VAE(enc,dec,latent)
		model.get_num_params()

	#TODO: Add loading function
	#if os.path.isfile(args.model):
	#    model.load(charset, args.model, latent_rep_size = args.latent_dim)

	#criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(), lr = args.lr)
	if DYN_LR:
		scheduler = ReduceLROnPlateau(optimizer, 'min', 
									factor = 0.2, 
									patience = 3,
									min_lr = 0.0001)

	dataloader = torch.utils.data.DataLoader(X_train, 
											batch_size=args.batch_size,
	                                        shuffle=False, 
	                                        num_workers=2,
	                                        drop_last = True)
	val_dataloader = torch.utils.data.DataLoader(X_test, 
											batch_size=args.batch_size,
	                                        shuffle=False, 
	                                        num_workers=2,
	                                        drop_last = True)

	print("###########################################################################")
	for epoch in range(args.epochs):
		epoch_loss = 0
		print("Epoch -- {}".format(epoch))
		for i, data in enumerate(dataloader):
			
			inputs = data.float()
			#inputs = inputs.reshape(batch_size, -1).float()
			optimizer.zero_grad()

			input_recon = model(inputs)
			latent_loss_val = latent_loss(model.z_mean, model.z_sigma)
			loss = F.binary_cross_entropy(input_recon, inputs, size_average=False) + latent_loss_val
			loss.backward()
			optimizer.step()
			epoch_loss += loss.item()

		print("Train Loss -- {:.3f}".format(epoch_loss/X_train.shape[0]))
		###Add 1 Image per Epoch for Visualisation
		data_point_sampled = random.randint(0,args.batch_size)

		print("Input -- "onehot_to_smiles(inputs[data_point_sampled], inv_dict))
		print("Output -- "onehot_to_smiles(model(inputs[data_point_sampled:data_point_sampled+1]), inv_dict))
				

		add_img(inputs[data_point_sampled], inv_dict, "Original_"+str(epoch))
		add_img(model(inputs[data_point_sampled:data_point_sampled+1]), inv_dict, "Recon_"+str(epoch))


		#####################Validation Phase
		epoch_loss_val = 0
		for i, data in enumerate(val_dataloader):
			
			inputs = data.float()
			#inputs = inputs.reshape(batch_size, -1).float()
			input_recon = model(inputs)
			latent_loss_val = latent_loss(model.z_mean, model.z_sigma)
			loss = F.binary_cross_entropy(input_recon, inputs, size_average=False) + latent_loss_val
			epoch_loss_val += loss.item()
		print("Validation Loss -- {:.3f}".format(epoch_loss_val/X_test.shape[0]))

		scheduler.step(epoch_loss_val)
		###Add 1 Image per Epoch for Visualisation
		#data_point_sampled = random.randint(0,args.batch_size)
		#add_img(inputs[data_point_sampled], inv_dict, "Original_"+str(epoch))
		#add_img(model(inputs[data_point_sampled:data_point_sampled+1]), inv_dict, "Recon_"+str(epoch))



		checkpoint = {'model': model.state_dict()}
		torch.save(checkpoint, args.save_loc+'/'+str(epoch)+'checkpoint.pth')


if __name__ == '__main__':
    main()

