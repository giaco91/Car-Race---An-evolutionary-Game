import sys
from copy import deepcopy

import numpy as np
from PIL import Image,ExifTags,ImageFilter,ImageOps, ImageDraw
import PIL
import numpy as np
import pickle

import matplotlib.pyplot as plt

from utils import *




#–---CAR-----
class Car():
	def __init__(self,v_max=5,grip=-0.5,m=1000,F_max=1000,size=0.08,model=2,backward=0.1,n_inputs=6,n_h=3,aerodynamic=1,mutate_physics=False):
		self.model=model
		self.v=0
		self.F_max=F_max
		self.m=m
		self.size=size
		self.aerodynamic=aerodynamic
		self.a_max=F_max/m
		self.v_max=v_max
		self.grip=grip#smaller or equal zero. zero is perfect grip. -1 is not good grip
		if mutate_physics:
			self.shape_mutation(mutation_rate=1.2)
		else:
			self.shape_mutation(mutation_rate=1)
		self.n_inputs=n_inputs
		self.n_h=n_h
		self.backward=backward#the capability to drive backward compared to forward
		self.a_weights=np.random.rand(self.n_h+1)-0.5
		self.c_weights=(np.random.rand(self.n_h+1)-0.5)
		self.h_weights=np.random.rand(self.n_inputs+2,self.n_h)-0.5#parametersharing of hidden representation

	def mutation(self,mutation_rate=1,shape_mutation=False):
		#mutation_rate can be any positive number
		#bias towards small parameter in order to counteract evolutionary drift into flat regions (sigmoid activations...)
		da=(np.random.rand(self.n_h+1)-0.5)*0.1*mutation_rate
		self.a_weights+=da*np.maximum(np.exp(-0.1*np.abs(self.a_weights)),-np.sign(self.a_weights*da))
		dc=(np.random.rand(self.n_h+1)-0.5)*0.1*mutation_rate
		self.c_weights+=dc*np.maximum(np.exp(-0.1*np.abs(self.c_weights)),-np.sign(self.c_weights*dc))
		dh=(np.random.rand(self.n_inputs+2,self.n_h)-0.5)*mutation_rate/self.n_h
		self.h_weights+=dh*np.maximum(np.exp(-0.1*np.abs(self.h_weights)),-np.sign(self.h_weights*dh))
		# print(np.linalg.norm(self.a_weights))
		# print(np.linalg.norm(self.c_weights))
		if shape_mutation:
			self.shape_mutation()

	def shape_mutation(self,mutation_rate=1.2):
		#mutationrate must be larger than one
		fac=np.random.rand(2)*(mutation_rate-1/mutation_rate)+1/mutation_rate
		self.grip=max(min(0,self.grip*fac[0]),-2)
		self.v_max=max(0,self.v_max*fac[1])
		self.transform_shape()


	def transform_shape(self):
		self.size=0.2*self.grip+0.4+self.v_max/10
		self.aerodynamic=1+self.v_max**2
		self.m=200+self.size/0.01
		self.F_max=1000*np.sqrt(self.aerodynamic)
		self.a_max=self.F_max/self.m

	def get_h(self,inputs):
		s=self.h_weights[0,:].copy()
		s+=self.h_weights[1,:]*self.v
		for i in range(inputs.shape[0]):
			s+=self.h_weights[i+2,:].copy()*inputs[i]
		return sigmoid(s)	

	def get_a(self,inputs):
		h=self.get_h(inputs)
		s=self.a_weights[0]
		for i in range(self.n_h):
			s+=self.a_weights[i+1]*h[i]
		a=sigmoid(s,bias=-0.5)*self.a_max
		if a<0:
			a*=self.backward
		if a>self.a_max-0.1:
			print('a_max reached')
		return a

	def get_c(self,inputs):
		h=self.get_h(inputs)
		s=self.c_weights[0]
		for i in range(self.n_h):
			s+=self.c_weights[i+1]*h[i]
		return sigmoid(s,bias=-0.5)*abs(self.v)*np.exp(abs(self.v)*self.grip)




