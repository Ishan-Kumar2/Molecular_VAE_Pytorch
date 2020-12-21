import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):
	def __init__(self,input_dim,hidden_dim,hidden_dim_2):
		super(Encoder,self).__init__()
		self.linear1 = nn.Linear(input_dim,hidden_dim)
		self.linear2 = nn.Linear(hidden_dim,hidden_dim_2)
		self.hidden_dim = hidden_dim
		self.hidden_dim_2 = hidden_dim_2
		
	def forward(self, x):
		z = F.relu(self.linear2(F.relu(self.linear1(x))))
		return z

class Decoder(torch.nn.Module):
	def __init__(self,input_dim,hidden_dim,hidden_dim_2):
		super(Decoder,self).__init__()
		self.linear1 = nn.Linear(hidden_dim_2,hidden_dim)
		self.linear2 = nn.Linear(hidden_dim,input_dim)

	def forward(self, z):
		x = F.relu(self.linear2(F.relu(self.linear1(z))))
		return x

class VAE(nn.Module):
	
	def __init__(self, encoder, decoder,latent_dim):
		super(VAE, self).__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.latent_dim = latent_dim

		self.hidden_dim_2 = self.encoder.hidden_dim_2
		self.hidden_dim = self.encoder.hidden_dim

		self._enc_mu = torch.nn.Linear(self.hidden_dim_2, self.latent_dim)
		self._enc_log_sigma = torch.nn.Linear(self.hidden_dim_2, self.latent_dim)
		

	def _sample_latent(self, h_enc):
		"""
		Return the latent normal sample z ~ N(mu, sigma^2)
		"""
		mu = self._enc_mu(h_enc)
		log_sigma = self._enc_log_sigma(h_enc)
		sigma = torch.exp(log_sigma)

		std_z = torch.randn(sigma.size()).float()
		
		self.z_mean = mu
		self.z_sigma = sigma

		return mu + sigma * std_z  # Reparameterization trick

	def forward(self, state):
		h_enc = self.encoder(state)
		z = self._sample_latent(h_enc)
		recon_x = self.decoder(z)
		return recon_x

	def get_num_params(self):
		print("Encoder--",sum(p.numel() for p in self.encoder.parameters() if p.requires_grad))
		print("Decoder--",sum(p.numel() for p in self.decoder.parameters() if p.requires_grad))
		print("Total--",sum(p.numel() for p in self.parameters() if p.requires_grad))

####################################################################################################
class Conv_Encoder(nn.Module):
	def __init__(self, vocab_len):
		super(Conv_Encoder,self).__init__()

		self.conv_1 = nn.Conv1d(120, 9, kernel_size=9)
		self.conv_2 = nn.Conv1d(9, 9, kernel_size=9)
		self.conv_3 = nn.Conv1d(9, 10, kernel_size=11)

		self.fc_1 = nn.Linear(10*(vocab_len-26),435)
		self.relu = nn.ReLU()

	def forward(self, x):
		batch_size = x.shape[0]
		x = self.relu(self.conv_1(x))
		x = self.relu(self.conv_2(x))
		x = self.relu(self.conv_3(x))
		x = x.reshape(batch_size, -1)
		x = self.relu(self.fc_1(x))
		return x
	
class GRU_Decoder(nn.Module):
	def __init__(self, vocab_size):
		super(GRU_Decoder,self).__init__()
		self.fc_1 = nn.Linear(292, 292)
		self.gru = nn.GRU(292, 501, 3, batch_first=True)
		self.fc_2 = nn.Linear(501, vocab_size)
		self.relu = nn.ReLU()
		self.softmax = nn.Softmax()
	def forward(self, z):
		batch_size = z.shape[0]
		z = self.relu(self.fc_1(z))
		z = z.unsqueeze(1).repeat(1, 120, 1)
		z_out, hidden = self.gru(z)
		z_out = z_out.contiguous().reshape(-1, z_out.shape[-1])
		x_recon = F.softmax(self.fc_2(z_out), dim=1)
		x_recon = x_recon.contiguous().reshape(batch_size, -1, x_recon.shape[-1])
		return x_recon

class Molecule_VAE(nn.Module):
	def __init__(self,encoder,decoder,device):
		super(Molecule_VAE,self).__init__()
		self.encoder = encoder
		self.decoder = decoder
		#self.latent_dim = latent_dim

		#self.hidden_dim_2 = self.encoder.hidden_dim_2
		#self.hidden_dim = self.encoder.hidden_dim
		self.relu = nn.ReLU()
		self.device = device

		self._enc_mu = nn.Linear(435,292)
		self._enc_log_sigma = nn.Linear(435,292)
		
	def _sample_latent(self, h_enc):
		"""
		Return the latent normal sample z ~ N(mu, sigma^2)
		"""
		mu = self._enc_mu(h_enc)
		log_sigma = self.relu(self._enc_log_sigma(h_enc))
		sigma = torch.exp(log_sigma)

		eps = 1e-2 * torch.randn_like(sigma).float().to(self.device)
		
		self.z_mean = mu
		self.z_sigma = sigma

		return mu + sigma * eps  # Reparameterization trick

	def forward(self, state):
		h_enc = self.encoder(state)
		z = self._sample_latent(h_enc)
		recon_x = self.decoder(z)
		return recon_x

	def get_num_params(self):
		print("Encoder--",sum(p.numel() for p in self.encoder.parameters() if p.requires_grad))
		print("Decoder--",sum(p.numel() for p in self.decoder.parameters() if p.requires_grad))
		print("Total--",sum(p.numel() for p in self.parameters() if p.requires_grad))



def latent_loss(z_mean, z_stddev):
	mean_sq = z_mean * z_mean
	stddev_sq = z_stddev * z_stddev
	#0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)
	return 0.5 * torch.mean(mean_sq + z_stddev - torch.log(stddev_sq) - 1)


if __name__ == '__main__':

	print("Checking Normal VAE")
	vocab_size = 71
	input_dim = 120 * vocab_size
	hidden_dim = 200
	hidden_2 = 120
	latent = 60

	enc = Encoder(input_dim,hidden_dim,hidden_2)
	dec = Decoder(input_dim,hidden_dim,latent)
	vae = VAE(enc,dec,latent)
	vae.get_num_params()
	
	criterion = nn.MSELoss()

	ex_input = torch.randn(1,120,71)
	ex_input = ex_input.reshape(1,-1)

	output = vae(ex_input)
	print("MSE LOSS",latent_loss(vae.z_mean,vae.z_sigma)+criterion(ex_input,output))
	print("Input Shape",ex_input.shape)
	print("Output Shape",output.shape)

	print("#######################################################################################")
	print("Checking Molecule VAE")

	enc = Conv_Encoder(vocab_size)
	dec = GRU_Decoder(vocab_size)

	model = Molecule_VAE(enc, dec, device)

	ex_input = torch.randn(1,120,71)
	model.get_num_params()
	output = model(ex_input)
	print("MSE LOSS",latent_loss(vae.z_mean,vae.z_sigma)+criterion(ex_input,output))
	print("Input Shape",ex_input.shape)
	print("Output Shape",output.shape)
	############################################################################################