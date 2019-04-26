#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:44:27 2019

@author: Veloc1ty
"""

import tensorflow as tf
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
tf.reset_default_graph()

loss_file = 'Epoch_Loss.txt'

# Load Data and process with discrete fourier transform
import librosa
data, sampling_rate = librosa.load('./audio_wav/0.wav')
data = librosa.core.stft(data, sampling_rate)

# Merge Channels
temp = np.zeros(shape=[len(data), len(data[0])*2])
for i in range(len(data[0])):
    temp[:, 2*i-1] = data[:, i].real
    temp[:, 2*i] = data[:, i].imag
    
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
temp = sc.fit_transform(temp)

#data_channel_1 = data.real
#data_channel_2 = data.imag

inputs = input_dim = output_dim = len(temp)
lr = 0.0001
batch_size = 512
epochs = 10000
z_size = 68

rate = tf.placeholder(dtype=tf.float32, shape=(), name='rate')

X = tf.placeholder(tf.float32, shape=[None, input_dim])
X_in = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='X')
Y    = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='Y')
Y_flat = tf.reshape(Y, shape=[-1, input_dim])

def encoder(X_in, rate):
    # Encoder for the music VAE
    with tf.variable_scope("encoder", reuse=None):
        # Encoder Architecture
        e = tf.layers.dense(X_in, 6800, activation='elu')
        e = tf.nn.dropout(e, rate)
        e = tf.layers.dense(e, 4600, activation='elu')
        e = tf.nn.dropout(e, rate)
        e = tf.layers.dense(e, 2400, activation='elu')
        e = tf.nn.dropout(e, rate)       
        e = tf.reshape(e, [-1, 2400])
        # Variational Autoencoder parameters
        mu     = tf.layers.dense(e, z_size, name="enc_mu")
        logvar = tf.layers.dense(e, z_size, name="enc_fc_log_var") 
        sigma  = tf.exp(logvar / 2.0)
        epsilon = tf.random_normal([batch_size, z_size])
        # Return value
        z = mu + sigma * epsilon 
        return z, mu, logvar

def decoder(sampled_z, rate):
    with tf.variable_scope("decoder", reuse=None):
        # Decoder Architecture
        d = tf.layers.dense(sampled_z, units=2400, activation='elu')
        d = tf.nn.dropout(d, rate)
        d = tf.layers.dense(d, 4400, activation='elu')
        d = tf.nn.dropout(d, rate)
        d = tf.layers.dense(d, 6800, activation='elu')
        d = tf.nn.dropout(d, rate)
        d = tf.layers.dense(d, output_dim, activation=None)
        X = tf.reshape(d, shape=[-1, output_dim])
        return X

sampled, mu, logvar = encoder(X_in, rate)
dec = decoder(sampled, rate)

array_loss = tf.reduce_sum(tf.squared_difference(dec, Y_flat), 1)
latent_loss = - 0.5 * tf.reduce_sum((1 + logvar - tf.square(mu) - tf.exp(logvar)), reduction_indices = 1)
loss = tf.reduce_mean(array_loss + latent_loss)

optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)
init=tf.global_variables_initializer()

sess = InteractiveSession(config=config)
sess.run(init)

sub_arr = temp[:, 60:572]
arr = np.array(sub_arr).reshape(batch_size, inputs)

sess.run(optimizer, feed_dict = {X_in: arr, Y: arr, rate: 0.2})

for epoch in range(epochs):
	epoch_loss = []
	print("Epoch: " + str(epoch))
	for i in range(batches):
		ch1_song, ch2_song, sample_rate = next_batch(i, batch_size, sess)
		total_songs = np.hstack([ch1_song, ch2_song])
		batch_loss = []
        
		for j in range(len(total_songs)):
			x_batch = total_songs[j]
			_, l = sess.run([training_op, loss], feed_dict={X:x_batch})
			batch_loss.append(l)
			print("Song loss: " + str(l))

		print("Curr Epoch: " + str(epoch) + " Curr Batch: " + str(i) + "/"+ str(batches))
		print("Batch Loss: " + str(np.mean(batch_loss)))
		epoch_loss.append(np.mean(batch_loss))

	print("Epoch Avg Loss: " + str(np.mean(epoch_loss)))
