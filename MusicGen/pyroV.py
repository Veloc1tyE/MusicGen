#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 12:06:02 2019

@author: Veloc1ty
"""

import torch
import pyro
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

inputs = outputs = 11026
epochs = 5000    
batch_size = 512
z_size = 68

assert pyro.__version__.startswith('0.3.3')
pyro.enable_validation(True)
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)
smoke_test = 'CI' in os.environ

import process_data
from glob import iglob
from sklearn.preprocessing import MinMaxScaler

file_arr = []
loss_file = 'Epoch_Loss.txt'


def process():
	for file in iglob('audio' +'/*.mp3'):
		file_arr.append(file)

process()

def get_data(file_arr):
    template, rate = process_data.get_data(file_arr)

    data = np.array(template[0])
    n = len(data[1])
    data = data.reshape(n, inputs)
    
    for idx in range(1, len(template)):
        sub_arr = np.array(template[idx])
        n = len(sub_arr[0])
        sub_arr = sub_arr.reshape(n, inputs)
        data=np.concatenate((data, sub_arr), axis=0)
    
    return data, rate

data, sample_rate = get_data(file_arr)

def rep(data):
    """converts data to length * num_channels than splits by real and complex part"""
    total = np.zeros(shape=[len(data)*2, len(data[0])])
    for i in range(len(data)):
        total[2*i-1, :] = data[i, :].real
        total[2*i, :] = data[i, :].imag
    return total

total = rep(data)

sc = MinMaxScaler((0, 1))
total = sc.fit_transform(total)
total = torch.tensor(total, dtype = torch.float32)

class Encoder(nn.Module):
    def __init__(self, z_size, var):
        super(Encoder, self).__init__()
        # setup the linear transformations used
        self.fc1 = nn.Linear(var, 6800)
        self.fc2 = nn.Linear(6800, 4600)
        self.fc3 = nn.Linear(4600, 2400)
        self.fc31 = nn.Linear(2400, z_size)
        self.fc32 = nn.Linear(2400, z_size)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # define the forward computation on the sample x
        h1 = self.sigmoid(self.fc1(x))
        h2 = self.sigmoid(self.fc2(h1))
        h3 = self.softplus(self.fc3(h2))
        # Latent Variables
        z_loc = self.fc31(h3)
        z_scale = torch.exp(self.fc32(h3))
        return z_loc, z_scale


class Decoder(nn.Module):
    def __init__(self, z_size, var):
        super(Decoder, self).__init__()
        # Define Decoder architecture
        self.fc1 = nn.Linear(z_size, 2400)
        self.fc2 = nn.Linear(2400, 4600)
        self.fc3 = nn.Linear(4600, 6800)
        self.fc31 = nn.Linear(6800, var)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, z):
        # Define the forward operation
        z1 = self.sigmoid(self.fc1(z))
        z2 = self.sigmoid(self.fc2(z1))
        z3 = self.softplus(self.fc3(z2))
        recon = self.sigmoid(self.fc31(z3))
        return recon
    

class VAE(nn.Module):
    def __init__(self, z_size=68, var=11026, use_cuda=False):
        # Initialise Encoder and Decoder networks
        super(VAE, self).__init__()
        self.encoder = Encoder(z_size, var)
        self.decoder = Decoder(z_size, var)
        
        if use_cuda:
            self.cuda()
        self.use_cuda = use_cuda
        self.z_size = z_size
        
    # Define the variational distribution q(z|x)
    def guide(self, x):
        # Register pytorch module 'encoder' with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # define our distribution q(z|x)
            z_loc, z_scale = self.encoder.forward(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            
    # Define the model p(x|z)p(z)
    def model(self, x):
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_size)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_size)))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            recon = self.decoder.forward(z)
            # Score against actual constructions
            pyro.sample("obs", dist.Bernoulli(recon).to_event(1), obs=x.reshape(-1, outputs))
            return recon
    
    # Define a function for reconstructing samples
    def reconstruct_sample(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        recon = self.decoder(z)
        return recon

# Run only for a single iteration for testing
NUM_EPOCHS = 1 if smoke_test else 50
TEST_FREQUENCY = 5

# Clear param store
pyro.clear_param_store()

# Setup the VAR
vae = VAE()
adam_args = {"lr": 1.0e-4}
optimizer = Adam(adam_args)

# setup the inference algorithm
svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())
train_elbo = []
test_elbo = []

arr = total[60:60+batch_size, :]
#arr = arr.cuda()

dataset = torch.utils.data.DataLoader(total, batch_size=512)
for batch in dataset:
    loss = 0
    loss += svi.step(batch)
    print(loss)
        

for epoch in range(epochs):
	epoch_loss = []
	print("Epoch: " + str(epoch))
	for i in range(len(total) // batch_size):
		batch = process_data.get_batch(i, batch_size, total)
		batch_loss = []
		_, l = sess.run([training_op, loss], feed_dict={X_in: batch, Y: batch, rate: 0.2})
		batch_loss.append(l)


arr = total[3, :] 

encoder = Encoder(68, 11026)
decoder = Decoder(68, 11026)
z_loc, z_scale = encoder(arr)
z = dist.Normal(z_loc, z_scale).sample()
sample = decoder(z)
        
    svi.step(arr)
        
        
    