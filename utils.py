import torch
import torch.nn as nn
import torchvision
import sys
from copy import deepcopy

import numpy as np
from PIL import Image,ExifTags,ImageFilter,ImageOps, ImageDraw
import PIL
import numpy as np
import pickle

import matplotlib.pyplot as plt


# np.random.seed(7)



def create_image(i, j):
	image = Image.new("RGB", (i, j), "white")
	return image

def sigmoid(s,bias=0):
	return 1/(1+np.exp(-s))+bias

def resize_to_height_ref(image,n_height):
    w,h=image.size
    return image.resize((n_height,round(n_height*h/w)),Image.ANTIALIAS)

def resize(image,W,H):
	return image.resize((W,H),Image.ANTIALIAS)

def reflect_y_axis(im):
	return ImageOps.mirror(im).rotate(180,expand=True)

def intersection_of_two_lines(p1,d1,p2,d2):
	#we define the lines: l1=p1+t[0]*d1 and l2=p2+t[1]*d2
	A=np.zeros((2,2))
	A[:,0]=d1
	A[:,1]=-d2
	det=np.linalg.det(A)
	if abs(det)<1e-10 or False:
		return np.array([None,None])
	else:
		A_inv=np.linalg.inv(A) 
		t=np.dot(A_inv,p2-p1)
		#s=p1+t[0]*d1
		return t

def rotation(alpha,v):
	c=np.cos(alpha)
	s=np.sin(alpha)
	R=np.array([[c,-s],[s,c]])
	return np.dot(R,v)

def get_angle(v1,v2):
	return np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))


def pack_sequences(data):
	#the data is a list of sequences. Every sequence is it self a list of data points.
    #the data must be preprocessed in order to use nn.utils.rnn.pack_padded_sequence
    #sequence size:(batchsize,max_seq_length,data_dim)
    n_sequences=len(data)
    data_point_dim=2+3#2 for the ds-vector and 3 for the measurment vector
    sequence_lengths=np.zeros(n_sequences).astype(int)
    for i in range(n_sequences):
    	sequence_lengths[i]=int(len(data[i]))
    sorted_idx=np.argsort(sequence_lengths)
    sorted_sequence_lengths=[]
    data_batch=torch.zeros(n_sequences,sequence_lengths[sorted_idx[-1]],5)
    for i in range(n_sequences):
    	data_batch[i,0:sequence_lengths[sorted_idx[-1-i]],:]=torch.from_numpy(np.asarray(data[sorted_idx[-1-i]])[:,np.asarray([0,1,2,4,5])])
    	sorted_sequence_lengths.append(sequence_lengths[sorted_idx[-1-i]])
    packed_data_batch = nn.utils.rnn.pack_padded_sequence(data_batch, sorted_sequence_lengths, batch_first=True)
    return packed_data_batch


def unpack_sequences(data_packed):
    #the inverse of pack sequences
    data, seq_lengths= nn.utils.rnn.pad_packed_sequence(data_packed, batch_first=True)
    return data,seq_lengths


