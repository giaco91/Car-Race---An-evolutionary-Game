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

def get_shifted_background(background_im,scaled_ds):
	fade_step=10
	W,H=background_im.size
	new_background_im=background_im.copy()
	px=background_im.load()
	new_px=new_background_im.load()
	for w in range(W):
		for h in range(H):
			if px[w,h]==(230,200,0):
				px[w,h]=(255,255,255)
			elif px[w,h]!=(255,255,255):
				px[w,h]=(min(px[w,h][0]+fade_step,255),min(px[w,h][1]+fade_step,255),min(px[w,h][2]+fade_step,255))
	for w in range(W):
		for h in range(H):
			shifted_w_idx=round(w+scaled_ds[0])
			shifted_h_idx=round(h+scaled_ds[1])
			if 0<=shifted_w_idx<=W-1 and 0<=shifted_h_idx<=H-1:
				new_px[w,h]=px[shifted_w_idx,shifted_h_idx]
	return new_background_im

def halu_step(shifted_background,scaled_ds,scaled_inputs,scores,prev_orientation,car_shape=None,size_car=16,path='cars/car_2.png',prev_input_coordinates=None):
	car_background_im=shifted_background.copy()
	draw = ImageDraw.Draw(car_background_im)
	car_im = Image.open(path)
	W,H=car_background_im.size
	W_2=int(W/2)
	H_2=int(H/2)
	dcirc=round(W/150)
	laser_rgb=(230,200,0)

	car_background_px=car_background_im.load()
	s1=rotation(np.pi/4,prev_orientation)
	s2=rotation(-np.pi/4,prev_orientation)
	c0=np.array([W_2+prev_orientation[0]*scaled_inputs[0],H_2+prev_orientation[1]*scaled_inputs[0]])
	c1=np.array([W_2+s1[0]*scaled_inputs[1],H_2+s1[1]*scaled_inputs[1]])
	c2=np.array([W_2+s2[0]*scaled_inputs[2],H_2+s2[1]*scaled_inputs[2]])

	draw.line([(W_2,H_2),tuple(c0)],fill=laser_rgb,width=1)
	draw.line([(W_2,H_2),tuple(c1)],fill=laser_rgb,width=1)
	draw.line([(W_2,H_2),tuple(c2)],fill=laser_rgb,width=1)
	
	draw.ellipse([tuple(c0-dcirc),tuple(c0+dcirc)], fill=(255,0,0), outline=None)
	draw.ellipse([tuple(c1-dcirc),tuple(c1+dcirc)], fill=(255,0,0), outline=None)
	draw.ellipse([tuple(c2-dcirc),tuple(c2+dcirc)], fill=(255,0,0), outline=None)
	if prev_input_coordinates is None:
		None
		# car_background_px[int(c0[0]),int(c0[1])]=(255,0,0)
		# car_background_px[int(c1[0]),int(c1[1])]=(255,0,0)
		# car_background_px[int(c2[0]),int(c2[1])]=(255,0,0)
	else:
		prev_input_coordinates[0].append(tuple(c0))
		prev_input_coordinates[1].append(tuple(c1))
		prev_input_coordinates[2].append(tuple(c2))
		# draw.line(prev_input_coordinates[0],fill=(255,100,100),width=1)
		# draw.line(prev_input_coordinates[1],fill=(255,100,100),width=1)
		# draw.line(prev_input_coordinates[2],fill=(255,100,100),width=1)

	no_car_background_im=car_background_im.copy()

	ds_norm=np.linalg.norm(scaled_ds)
	if ds_norm==0:
		current_orientation=np.array([1,0])
	else:
		current_orientation=scaled_ds/ds_norm
		c0-=scaled_ds
		c1-=scaled_ds
		c2-=scaled_ds
	current_input_coordinates=[[tuple(c0)],[tuple(c1)],[tuple(c2)]]
	phi=np.arctan(current_orientation[0]/(current_orientation[1]+1e-10))*360/(2*np.pi)+180#need to add 180 because show() has y-coordinate southwards
	if scaled_ds[1]<0:
		phi+=180
	if car_shape is not None:
		if size_car<=0:
			print('size_car is smaller or equal to zero: '+str(size_car))
		if int(size_car*car_shape)<=0:
			print('size_car times car_shape is smaller or equal to zero: '+str(int(size_car*car_shape)))
		car_im=resize(car_im,size_car,int(size_car*car_shape))
	else:
		car_im=resize_to_height_ref(car_im,size_car)
	car_im=car_im.rotate(phi, expand=True)
	w,h=car_im.size
	w_2=int(w/2)
	h_2=int(h/2)
	car_im_px=car_im.load()
	for i in range(w):
		for j in range(h):
			sp=car_im_px[i,j][0]+car_im_px[i,j][1]+car_im_px[i,j][2]
			idx_w=W_2-w_2+i
			idx_w=int(min(max(idx_w,0),W-1))
			idx_h=H_2-h_2+j
			idx_h=int(min(max(idx_h,0),H-1))
			if 10<sp<750 and 0<=idx_w<W and 0<=idx_h<H:
				car_background_px[idx_w,idx_h]=car_im_px[i,j]
	return car_background_im, no_car_background_im,current_input_coordinates

def get_reg_data(data):
	l_tot=0
	for i in range(len(data)):
		l_tot+=len(data[i])
	x=torch.zeros(l_tot,5)
	t=torch.zeros(l_tot,4)
	current_l=0
	for i in range(len(data)):
		x[current_l:current_l+len(data[i]),:]=torch.from_numpy(np.asarray(data[i])[:,np.array([0,1,2,4,5])])
		t[current_l:current_l+len(data[i]),:4]=torch.from_numpy(np.asarray(data[i]))[:,:4]
		current_l+=len(data[i])
	return x,t

def get_loss(m_hat,r_hat,m,r,batchSize,seq_lengths):
	loss=0
	for bs in range(batchSize):
		dm_bs=m_hat[bs,:seq_lengths[bs],:]-m[bs,:seq_lengths[bs],:]
		dr_bs=r_hat[bs,:seq_lengths[bs],:]-r[bs,:seq_lengths[bs],:]
		loss+=(torch.sum(torch.mul(dm_bs,dm_bs))+torch.sum(torch.mul(dr_bs,dr_bs)))/int(seq_lengths[bs])
	return loss

def get_reg_loss(m_hat,r_hat,m,r):
	L=m_hat.size(0)
	d_m=m_hat-m
	d_r=r_hat-r
	loss=(torch.sum(torch.mul(d_m,d_m))+torch.sum(torch.mul(d_r,d_r)))/L
	return loss

def train_environment_model(environment_model,optimizer,data,n_epochs=1000,save_path='dream_models/environment_model.pkl',print_every=200):
	batch_size=len(data)
	for i in range(n_epochs):
		optimizer.zero_grad()
		packed_sequences=pack_sequences(data)
		out_m, out_r, h_rare=environment_model(packed_sequences,batch_size)
		unpacked_sequences,seq_lengths=unpack_sequences(packed_sequences)
		m=unpacked_sequences[:,:,0:3]
		r=unpacked_sequences[:,:,3].unsqueeze(2)
		loss=get_loss(out_m,out_r,m,r,batch_size,seq_lengths)
		loss.backward()
		optimizer.step()
		if i%print_every==0:
			print('rnn loss: '+str(loss.item()))
	torch.save({'model_state': environment_model.state_dict(),'optimizer_state': optimizer.state_dict()}, save_path)
	return environment_model

def train_regression_model(reg_model,optimizer_reg,data,n_epochs=1000,save_path='dream_models/regression_model.pkl',print_every=200):
	x,t=get_reg_data(data)
	for i in range(n_epochs):
		optimizer_reg.zero_grad()
		m,r=reg_model(x)
		loss_reg=get_reg_loss(m,r,t[:,0:3],t[:,3].unsqueeze(1))
		loss_reg.backward()
		optimizer_reg.step()
		if i%print_every==0:
			print('reg loss: '+str(loss_reg.item()))
	torch.save({'model_state': reg_model.state_dict(),'optimizer_state': optimizer_reg.state_dict()}, save_path)



