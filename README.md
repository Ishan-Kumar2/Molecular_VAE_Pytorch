# A PyTorch implementation of Molecular VAE paper

PyTorch implementation of the paper **"Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules"**\
Link to Paper - [arXiv](https://arxiv.org/abs/1610.02415)

## Getting the Dataset
For this work I have used the ChEMBL Dataset which can be found [here](https://www.ebi.ac.uk/chembl/) \
\
Since the whole dataset has over 16M datapoints, I have decided to use a subset of that data.
To get the subset you can either use the train, val data present in ``/data``
or run the ``data_prep.py`` file as -
`python data_prep.py /path/to/downloaded_data col_name_smiles /save/path 50000` \
\
This will prepare 2 CSV files `/save/path_train.csv` and `/save/path_val.csv` both of length 50k and having randomly shuffled datapoints.

## Training the Network
To train the network use the `main.py` file

To Run the Papers Model (Conv Encoder and GRU Decoder)\
`python main.py ./data/chembl_500k_train ./data/chembl_500k_train ./Save_Models/ --epochs 100 --model_type mol_vae --latent_dim 290 --batch_size 512 --lr 0.0001`\
Latent Dim has default value 292 which is the value used in the original Paper

To Run a VAE with Fully Connected layers in both Encoder Decoder\
``python main.py ./data/bbbp.csv ./Save_Models/ --epochs 1 --model_type fc --latent_dim 100 --batch_size 20 --lr 0.0001``


## Results

The Train and Validation Losses where tracked for run of 100 epochs 
![Loss graphs](/Sample_imgs/graph_loss.png)
