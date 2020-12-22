# A PyTorch implementation of Molecular VAE paper

PyTorch implementation of the paper **"Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules"**\
Link to Paper - [arXiv](https://arxiv.org/abs/1610.02415)

## Getting the Repo
To clone the repo on your machine run - \
`git clone https://github.com/Ishan-Kumar2/Molecular_VAE_Pytorch.git` \
The Structure of the Repo is as follows -\
`data_prep.py`- For Getting the Data in CSV format and splitting into specifed sized Train and Val \
`main.py` - Running the model \
`model.py` - Defines the Architecture of the Model \
`utils.py` - Various useful functions for encoding and decoding the data \


## Getting the Dataset
For this work I have used the ChEMBL Dataset which can be found [here](https://www.ebi.ac.uk/chembl/) \
\
Since the whole dataset has over 16M datapoints, I have decided to use a subset of that data.
To get the subset you can either use the train, val data present in ``/data``
or run the ``data_prep.py`` file as - \
`python data_prep.py /path/to/downloaded_data col_name_smiles /save/path 50000` \
\
This will prepare 2 CSV files `/save/path_train.csv` and `/save/path_val.csv` both of length 50k and having randomly shuffled datapoints.

## Training the Network
To train the network use the `main.py` file

To Run the Papers Model (Conv Encoder and GRU Decoder)\
`python main.py ./data/chembl_500k_train ./data/chembl_500k_val ./Save_Models/ --epochs 100 --model_type mol_vae --latent_dim 290 --batch_size 512 --lr 0.0001`\
Latent Dim has default value 292 which is the value used in the original Paper

To Run a VAE with Fully Connected layers in both Encoder Decoder\
``python main.py ./data/bbbp.csv ./Save_Models/ --epochs 1 --model_type fc --latent_dim 100 --batch_size 20 --lr 0.0001``


## Results

The Train and Validation Losses where tracked for Training and Validation epochs

** Using Latent Dim = 292 (As in the Paper) ** 
![Loss graphs](/Sample_imgs/graph_loss_200.png)


** Using Latent Dim = 392 ** 
![Loss graphs](/Sample_imgs/Graph_loss_392.png)


### Sample Outputs

*Input* - CC(NC(=O)C)OC(=O)c1ccc(cc1)[N+](=O)[O-] \
*Output* - CC(C))CCCCCCCCOOOcc1cccccc1[N+](=O)[O-] 

*Input* - COC(=O)[C@H](CCCN=C(N)N)NC(=O)[C@H](Cc1c[nH]c2ccccc12)NC(=O)C3CCCCC3 \
*Output* - CC(=))CCCCCCCCCCCCCCCOO)))cccccccccccccccc)CCCCCCCCCCCCC3 

*Input* - c1ccsc1 \
Output -  c1cccc1 

*Input* - Cl.NCc1cc(Cl)cc(Cl)c1 \
*Output* -Cl.CCc1cc((Cl)ClClcc1 

*Input* - CC(C)CCOc1ccc(Cl)cc1C(=C)n2cncn2 \
*Output*- CC(C)CO)c1cccccccccccccccnCCnn3 

*Input*- CCCCCCCCCCc1cccc(O)c1C(=O)O \
*Output*-CCCCCCCCCCc1ccccccc)CC(O))O 
