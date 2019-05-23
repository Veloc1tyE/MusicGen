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
import process_data
from glob import iglob
import librosa
from sklearn.preprocessing import MinMaxScaler

config = ConfigProto()
config.gpu_options.allow_growth = True
tf.reset_default_graph()

file_arr = []
loss_file = 'Epoch_Loss.txt'


def process():
	for file in iglob('audio' +'/*.mp3'):
		file_arr.append(file)

process()

#data_channel_1 = data.real
#data_channel_2 = data.imag

inputs = outputs = 11026
epochs = 5000    
batch_size = 512
z_size = 68

rate = tf.placeholder(dtype=tf.float32, shape=(), name='rate')

X = tf.placeholder(tf.float32, shape=[None, inputs])
X_in = tf.placeholder(dtype=tf.float32, shape=[None, inputs], name='X')
Y    = tf.placeholder(dtype=tf.float32, shape=[None, inputs], name='Y')
Y_flat = tf.reshape(Y, shape=[-1, inputs])

def encoder(X_in, rate):
    # Encoder for the music VAE
    with tf.variable_scope("encoder", reuse=None):
        # Encoder Architecture
        e1 = tf.layers.dense(X_in, 6800, activation='tanh')
        e = tf.nn.dropout(e1, rate=rate)
        e2 = tf.layers.dense(e, 4600, activation='tanh')
        e = tf.nn.dropout(e2, rate=rate)
        e3 = tf.layers.dense(e, 2400, activation='tanh')
        # Variational Autoencoder parameters
        mn     = tf.layers.dense(e3, z_size, name="enc_mu")
        sd     = 0.5 * tf.layers.dense(e3, z_size, name='enc_sd')  
        epsilon = tf.random_normal(tf.stack([tf.shape(e3)[0], n_latent]))
        # Return value
        z  = mn + tf.multiply(epsilon, tf.exp(sd))
        return z, mn, sd, e1, e2, e3

def decoder(sampled_z, rate):
    with tf.variable_scope("decoder", reuse=None):
        # Decoder Architecture
        d = tf.layers.dense(sampled_z, units=2400, activation='tanh')
        d = e3 + d
        d = tf.nn.dropout(d, rate=rate)
        d = tf.layers.dense(d, 4600, activation='tanh')
        d = e2 + d
        d = tf.nn.dropout(d, rate=rate)
        d = tf.layers.dense(d, 6800, activation='tanh')
        d = e1 + d
        d = tf.layers.dense(d, outputs, activation='tanh')
        X = tf.reshape(d, shape=[-1, outputs])
        return X
    
def get_data(file_arr):
    template, rate = process_data.get_data(file_arr)

    data = np.array(template[0])
    n = len(data[1])
    data = data.reshape(n, inputs)

    # Ch2
    #ch2_arr = np.array(ch2[0])
    #n2 = len(ch2_arr[1])
    #ch2_arr = ch2_arr.reshape(n2, inputs)
    
    for idx in range(1, len(template)):
        sub_arr = np.array(template[idx])
        n = len(sub_arr[0])
        sub_arr = sub_arr.reshape(n, inputs)
        data=np.concatenate((data, sub_arr), axis=0)
    
    #for idx in range(1, len(ch2)):
     #   sub_arr2 = np.array(ch2[idx])
      #  n2 = len(sub_arr2[1])
       # sub_arr2 = sub_arr2.reshape(n2, inputs)
        #ch2_arr=np.concatenate((ch2_arr, sub_arr2), axis=0)
    return data, rate

z, mn, sd, e1, e2, e3  = encoder(X_in, rate)
dec = decoder(z, rate)

MSE = tf.reduce_sum(tf.squared_difference(dec, Y_flat), 1)
latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
loss = tf.reduce_mean(latent_loss + MSE)
optimizer = tf.train.AdamOptimizer(0.0001)
training_op = optimizer.minimize(loss)
saver = tf.train.Saver()

init=tf.global_variables_initializer()
sess = InteractiveSession(config=config)
sess.run(init)

# Data loading, merging and testing
data, sample_rate = get_data(file_arr)

def rep(data):
    """converts data to length * num_channels than splits by real and complex part"""
    total = np.zeros(shape=[len(data)*2, len(data[0])])
    for i in range(len(data)):
        total[2*i-1, :] = data[i, :].real
        total[2*i, :] = data[i, :].imag
    return total

def invrep(d):
    """Reshapes to num_channels * length than merges by complex part"""
    d = sc.inverse_transform(d)
    ch1 = []
    ch2 = []
    for i in range(len(d)):
        if i % 2:
            ch1.append(d[i, :])
        else:
            ch2.append(d[i, :])
    ch1 = np.array(ch1)
    
    toxic = ch1[len(ch1)-1, :]
    fuck = ch1[0:len(ch1)-1, :]
    
    ch1 = np.vstack((toxic, fuck))    
    ch2 = np.array(ch2)
    result = ch1 + 1j*ch2
    result = result.reshape(len(result[1]), len(result))
    return result
    

total = rep(data)

sc = MinMaxScaler((-0.95, 0.95))
total = sc.fit_transform(total)

#arr = total[60:60+batch_size, :]
#_, l = sess.run([training_op, loss], feed_dict = {X_in: arr, Y: arr, rate: 0.2})
#print("Batch loss: " + str(l))

for epoch in range(epochs):
	epoch_loss = []
	print("Epoch: " + str(epoch))
	for i in range(len(total) // batch_size):
		batch = process_data.get_batch(i, batch_size, total)
		batch_loss = []
		_, l = sess.run([training_op, loss], feed_dict={X_in: batch, Y: batch, rate: 0.2})
		batch_loss.append(l)

	print("Curr Epoch: " + str(epoch))
	print("Batch Loss: " + str(np.mean(batch_loss)))
	epoch_loss.append(np.mean(batch_loss))

print("Epoch Avg Loss: " + str(np.mean(epoch_loss)))
save_path = saver.save(sess, "./Model/model1.ckpt") 
print("Model saved in path: %s" % save_path)

del total
del data
del batch

matrix, rate1 = librosa.load(file_arr[1])
matrix = librosa.core.stft(matrix, rate1)
matrix = np.array(matrix)
n = len(matrix[1])
prediction = matrix.reshape(n, inputs)
prediction = rep(prediction)
prediction = sc.transform(prediction)

#test = invrep(prediction)


d = sess.run(dec, feed_dict={X_in: prediction[0:512, :], rate: 0.0})
for i in range(1, len(prediction) // batch_size):
	batch = process_data.get_batch(i, batch_size, prediction)
	new = sess.run(dec, feed_dict={X_in: batch, rate: 0.0})
	d = np.concatenate((new, d))

pred = invrep(d)
generated_song = librosa.istft(pred)
librosa.output.write_wav('./output/generated_song.wav', generated_song, sample_rate)
