from model import *
from utils import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import argparse
import random


NUM_EPOCHS = 30
BATCH_SIZE = 256
LATENT_DIM = 292
RANDOM_SEED = 42
LR = 0.0001
DYN_LR = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular VAE network')
    parser.add_argument('data', type=str, help='Path to the dataset and name')

    parser.add_argument('val_data', type=str, help='Path to the Validation dataset and name')

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
#torch.set_printoptions(threshold=10_000)
def main():
    args = get_arguments()

    data = pd.read_csv(args.data)

    smiles = data[SMILES_COL_NAME]
    #labels = np.array(data['p_np'])
    labels = np.zeros((len(smiles),1))
    print("Example Smiles",smiles[0:10])

    ##Building the Vocab from DeepChem's Regex
    vocab, inv_dict = build_vocab(data)
    vocab_size = len(vocab)
    print(vocab)
    print(vocab.items())
    ##Converting to One Hot Vectors
    data = make_one_hot(data[SMILES_COL_NAME],vocab)
    print("Input Data Shape",data.shape)
    
    ##Checking ratio for Classification Datasets
    #print("Ratio of Classes")
    #get_ratio_classes(labels)

    data_val = pd.read_csv(args.val_data)
    smiles_val = data_val[SMILES_COL_NAME]
    labels_val = np.zeros((len(smiles),1))
    data_val = make_one_hot(data_val[SMILES_COL_NAME],vocab)

    X_train = data 
    X_test = data_val
    y_train = labels 
    y_test = labels_val
    ##To oversample datasets if dataset is imbalanced
    oversample = False
    if oversample:
        print(data.shape,labels.shape)
        data,labels = oversample(data,labels)
        print("After Over Sampling")
        get_ratio_classes(labels_oversampled)

    ##Train Test Split
    
    #X_train, X_test, y_train, y_test = split_data(data,labels)
    #print("Train Data Shape--{} Labels Shape--{} ".format(X_train.shape,y_train.shape))
    #print("Test Data Shape--{} Labels Shape--{} ".format(X_test.shape,y_test.shape))


    use_vae = args.model_type == 'mol_vae'
    if use_vae:
        ##Using Molecular VAE Arch as in the Paper with Conv Encoder and GRU Decoder
        enc = Conv_Encoder(vocab_size).to(device)
        dec = GRU_Decoder(vocab_size,latent_dim).to(device)
        model = Molecule_VAE(enc, dec,device,latent_dim).to(device)
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
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
                                    factor = 0.8, 
                                    patience = 3,
                                    min_lr = 0.0001)

    dataloader = torch.utils.data.DataLoader(X_train, 
                                            batch_size=args.batch_size,
                                            shuffle=True, 
                                            num_workers=6,
                                            drop_last = True)
    val_dataloader = torch.utils.data.DataLoader(X_test, 
                                            batch_size=args.batch_size,
                                            shuffle=True, 
                                            num_workers=6,
                                            drop_last = True)

    best_epoch_loss_val = 100000
    x_train_data_per_epoch = X_train.shape[0] - X_train.shape[0]%args.batch_size
    x_val_data_per_epoch = X_test.shape[0] - X_test.shape[0]%args.batch_size
    print("Div Quantities",x_train_data_per_epoch,x_val_data_per_epoch)
    print()
    print("###########################################################################")
    for epoch in range(args.epochs):
        epoch_loss = 0
        print("Epoch -- {}".format(epoch))
        #data_points_done = 0
        for i, data in enumerate(dataloader):
            
            inputs = data.float().to(device)
            #inputs = inputs.reshape(batch_size, -1).float()
            optimizer.zero_grad()

            input_recon = model(inputs)
            latent_loss_val = latent_loss(model.z_mean, model.z_sigma)
            loss = F.binary_cross_entropy(input_recon, inputs, size_average=False) + latent_loss_val
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            #data_points_done += data.shape[0]

        print("Train Loss -- {:.3f}".format(epoch_loss/x_train_data_per_epoch))
        ###Add 1 Image per Epoch for Visualisation
        data_point_sampled = random.randint(0,args.batch_size-1)

        #print("INPUT ARGMAX",inputs[data_point_sampled].reshape(1, 120, len(vocab)))

        print("INPUT",inputs[data_point_sampled])
        print("OUTPUT",input_recon[data_point_sampled].reshape(1, 120, len(vocab)))

        print("Input -- ",onehot_to_smiles(inputs[data_point_sampled].reshape(1, 120, len(vocab)).cpu().detach(), inv_dict))
        print("Output -- ",onehot_to_smiles(input_recon[data_point_sampled].reshape(1, 120, len(vocab)).cpu().detach(), inv_dict))
        
        #print("len",len(onehot_to_smiles(inputs[data_point_sampled].cpu().detach().numpy(), inv_dict)))       

        #add_img(inputs[data_point_sampled], inv_dict, "Original_"+str(epoch))
        #ladd_img(model(inputs[data_point_sampled:data_point_sampled+1]), inv_dict, "Recon_"+str(epoch))


        #####################Validation Phase
        epoch_loss_val = 0
        for i, data in enumerate(val_dataloader):
            
            inputs = data.to(device).float()
            #inputs = inputs.reshape(batch_size, -1).float()
            input_recon = model(inputs)
            latent_loss_val = latent_loss(model.z_mean, model.z_sigma)
            loss = F.binary_cross_entropy(input_recon, inputs, size_average=False) + latent_loss_val
            epoch_loss_val += loss.item()
        print("Validation Loss -- {:.3f}".format(epoch_loss_val/x_val_data_per_epoch))
        print()
        scheduler.step(epoch_loss_val)
        ###Add 1 Image per Epoch for Visualisation
        #data_point_sampled = random.randint(0,args.batch_size)
        #add_img(inputs[data_point_sampled], inv_dict, "Original_"+str(epoch))
        #add_img(model(inputs[data_point_sampled:data_point_sampled+1]), inv_dict, "Recon_"+str(epoch))



        checkpoint = {'model': model.state_dict(),
                    'dict':vocab,
                    'inv_dict':inv_dict,
                    }

        if epoch_loss_val < best_epoch_loss_val:
            torch.save(checkpoint, args.save_loc+'/'+str(epoch)+'checkpoint.pth')
        best_epoch_loss_val = min(epoch_loss_val, best_epoch_loss_val)
    #evaluate(model, X_train, vocab, inv_dict)


def evaluate(model, X_train, vocab, inv_dict):
    print("IN EVALUATION PHASE")
    pretrained = torch.load('./Save_Models/189checkpoint.pth', map_location=lambda storage, loc: storage)
    #torch.load('./Save_Models/189checkpoint.pth',map_location=torch.device('cpu'))
    dataloader = torch.utils.data.DataLoader(X_train, 
                                            batch_size=1,
                                            shuffle=False, 
                                            num_workers=2,
                                            drop_last = True)
    for i, data in enumerate(dataloader):
        inputs = data.float().to(device)
        input_recon = model(inputs)        
        print(i)
        print("Input -- ",onehot_to_smiles(inputs[0].reshape(1, 120, len(vocab)).cpu().detach(), inv_dict))
        print("Output -- ",onehot_to_smiles(input_recon[0].reshape(1, 120, len(vocab)).cpu().detach(), inv_dict))
        print()


if __name__ == '__main__':
    
    main()
