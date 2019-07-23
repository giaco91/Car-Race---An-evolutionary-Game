import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import sys
from copy import deepcopy

import numpy as np
from PIL import Image,ExifTags,ImageFilter,ImageOps, ImageDraw
import PIL
import numpy as np
import pickle
from utils import *
import matplotlib.pyplot as plt

class Environment_model(nn.Module):
    def __init__(self,h_dim=7,n_layers=1,n_mixtures=2,depth=1,input_size=5):
    	super(Environment_model, self).__init__()
    	self.h_dim=h_dim
    	self.n_direction=1
    	self.depth=depth
    	self.n_mixtures=n_mixtures
    	self.n_layers=1
    	self.rnn = nn.GRU(input_size=input_size,hidden_size=h_dim,num_layers=self.n_layers,batch_first=True,bidirectional=False)
    	self.h0 = nn.Linear(1, h_dim,bias=False)
    	self.dummy_input=torch.torch.FloatTensor([1])
    	self.hidden_to_out=nn.Sequential()
    	if depth>=2:
    		self.hidden_to_out.add_module("layer "+str(1),nn.Sequential(nn.Linear(h_dim,h_dim*2),nn.ReLU()))
    		self.out_m = nn.Linear(h_dim*2,3*3*n_mixtures)
    		self.out_r = nn.Linear(h_dim*2,1*3*n_mixtures)
    	else:
    		self.out_m = nn.Linear(h_dim,3*3*n_mixtures)
    		self.out_r = nn.Linear(h_dim,1*3*n_mixtures)
    	for i in range(1,depth-1):
    		self.hidden_to_out.add_module("layer "+str(i+1),nn.Sequential(nn.Linear(h_dim*2,h_dim*2),nn.ReLU()))

    def get_h_init(self,batch_size):
    	dummy_input=self.dummy_input.repeat(batch_size,1)
    	return self.h0(dummy_input).unsqueeze(0)


    def forward(self, x,batch_size,h_init=None):
    	#----rnn encoder
    	if h_init is None:
    		h_init=self.get_h_init(batch_size)#size: (n_direction,batch_size,h_dim)
    	all_h,h_last = self.rnn(x,h_init)#will be initialized with h_init
    	unpacked_all_h,seq_len=unpack_sequences(all_h)
    	out_m=torch.zeros(batch_size,seq_len[0],3*3*self.n_mixtures)
    	out_r=torch.zeros(batch_size,seq_len[0],1*3*self.n_mixtures)
    	for bs in range(batch_size):
    		out=self.hidden_to_out(unpacked_all_h[bs,:,:])
    		out_m[bs,:,:]=self.out_m(out)
    		out_r[bs,:,:]=self.out_r(out)
    		# out_m[bs,:,:]=self.out_m(unpacked_all_h[bs,:,:])
    		# out_r[bs,:,:]=self.out_r(unpacked_all_h[bs,:,:])
    	return out_m, out_r, h_last

class Regression_model(nn.Module):
    def __init__(self,n_layers=1):
    	super(Regression_model, self).__init__()
    	self.reg = nn.Sequential()
    	for i in range(n_layers-1):
    		self.reg.add_module("layer "+str(i+1),nn.Sequential(nn.Linear(5,5),nn.LeakyReLU()))
    		print('added layer '+str(i))
    	self.out_m = nn.Linear(5,3)
    	self.out_r = nn.Linear(5,1)

    def forward(self,x,h_init=None):
    	h=self.reg(x)
    	m=self.out_m(h)
    	r=self.out_r(h)
    	return m,r







