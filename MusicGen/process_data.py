#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 12:25:37 2019

@author: Veloc1ty
"""

import librosa
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops
from tensorflow.contrib import ffmpeg
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from scipy.fftpack import rfft, irfft
from glob import iglob
from pydub import AudioSegment

DATA_FILES_MP3 = 'audio'
DATA_FILES_WAV = 'audio_wav'
file_arr = []
curr_batch = 0

def convert_mp3_to_wav():
	index = 0
	for file in iglob(DATA_FILES_MP3 + '/*.mp3'):
		mp3_to_wav = AudioSegment.from_mp3(file)
		mp3_to_wav.export(DATA_FILES_WAV+'/'+str(index)+'.wav', format='wav')
		index += 1

def process_wav():
	for file in iglob(DATA_FILES_WAV +'/*.wav'):
		file_arr.append(file)

def get_batch(curr_batch, batch_size, total):
	if ( (curr_batch*batch_size + batch_size) >= (len(total)) ):
		curr_batch = 0

	start_position = curr_batch * batch_size
	end_position = start_position + batch_size
	batch = total[start_position:end_position, :]

	return batch

def get_data(file_arr):
    #ch1 = []
    #ch2 = []
    table = []
    
    """if ( (curr_batch*songs_per_batch) + songs_per_batch >= len(file_arr) ):
            curr_batch = 0"""
    start_position = 0
    end_position = len(file_arr)
    
    for idx in range(start_position, end_position):
        
        data, rate = librosa.load(file_arr[idx])
        data = librosa.core.stft(data, rate)
        data = np.array(data)
        
        table.append(data)
    
    return table, rate    
        
    

def save_to_wav(audio_arr_ch1, audio_arr_ch2, sample_rate, original_song_ch1, original_song_ch2, idty, folder, sess):
	audio_arr_ch1 = irfft(np.hstack(np.hstack(audio_arr_ch1)))
	audio_arr_ch2 = irfft(np.hstack(np.hstack(audio_arr_ch2)))

	original_song_ch1 = irfft(np.hstack(np.hstack(original_song_ch1)))
	original_song_ch2 = irfft(np.hstack(np.hstack(original_song_ch2)))
	
	original_song = np.hstack(np.array((original_song_ch1, original_song_ch2)).T)
	audio_arr = np.hstack(np.array((audio_arr_ch1, audio_arr_ch2)).T)

	print(original_song)
	w = np.linspace(0, sample_rate, len(audio_arr))
	w = w[0:len(audio_arr)]
	plt.figure(1)

	plt.plot(w, original_song)
	plt.savefig(str(folder) + '/original.png')
	plt.plot(w, audio_arr)
	plt.xlabel('sample')
	plt.ylabel('amplitude')
	plt.savefig(str(folder) + '/compressed' + str(idty) + '.png')
	plt.clf()	

	cols = 2
	rows = math.floor(len(audio_arr)/2)
	audio_arr = audio_arr.reshape(rows, cols)
	original_song = original_song.reshape(rows, cols)

	wav_encoder = ffmpeg.encode_audio(
		audio_arr, file_format='wav', samples_per_second=sample_rate)
	wav_encoder_orig = ffmpeg.encode_audio(
		original_song, file_format='wav', samples_per_second=sample_rate)

	wav_file = sess.run(wav_encoder)
	wav_orig = sess.run(wav_encoder_orig)
	open(str(folder)+'/out.wav', 'wb').write(wav_file)
	open(str(folder)+'/wav_orig.wav', 'wb').write(wav_orig)

