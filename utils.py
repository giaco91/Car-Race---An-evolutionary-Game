import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
import sys
from copy import deepcopy
import math
import numpy as np
from PIL import Image,ExifTags,ImageFilter,ImageOps, ImageDraw, ImageFont
import PIL
import numpy as np
import pickle
import os 
import cv2 

import matplotlib.pyplot as plt


# np.random.seed(7)



def create_image(i, j):
	image = Image.new("RGB", (i, j), "white")
	return image

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

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
	alpha=np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
	if math.isnan(alpha):
		alpha=0#I tont know how else to solve it. if v1 is almost equal to v2 something strange happens and alpha becomes none
	return alpha

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def pack_sequences(data,with_alpha=False):
	#the data is a list of sequences. Every sequence is it self a list of data points.
    #the data must be preprocessed in order to use nn.utils.rnn.pack_padded_sequence
    #sequence size:(batchsize,max_seq_length,data_dim)
    n_sequences=len(data)
    sequence_lengths=np.zeros(n_sequences).astype(int)
    for i in range(n_sequences):
    	sequence_lengths[i]=int(len(data[i]))
    sorted_idx=np.argsort(sequence_lengths)
    sorted_sequence_lengths=[]
    data_batch=torch.zeros(n_sequences,sequence_lengths[sorted_idx[-1]],5)
    for i in range(n_sequences):
    	if with_alpha:
    		data_idx=np.asarray([0,1,2,4,5,6])
    	else:
    		data_idx=np.asarray([0,1,2,4,5])
    	data_batch[i,0:sequence_lengths[sorted_idx[-1-i]],:]=torch.from_numpy(np.asarray(data[sorted_idx[-1-i]])[:,data_idx])
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
	if prev_input_coordinates is not None:
		prev_input_coordinates[0].append(tuple(c0))
		prev_input_coordinates[1].append(tuple(c1))
		prev_input_coordinates[2].append(tuple(c2))

	no_car_background_im=car_background_im.copy()

	ds_norm=np.linalg.norm(scaled_ds)
	if ds_norm==0:
		current_orientation=prev_orientation
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

# def get_ls_loss(m_hat,r_hat,m,r,batchSize,seq_lengths):
# 	loss=0
# 	for bs in range(batchSize):
# 		dm_bs=m_hat[bs,:seq_lengths[bs],:]-m[bs,:seq_lengths[bs],:]
# 		dr_bs=r_hat[bs,:seq_lengths[bs],:]-r[bs,:seq_lengths[bs],:]
# 		loss+=(torch.sum(torch.mul(dm_bs,dm_bs))+torch.sum(torch.mul(dr_bs,dr_bs)))/int(seq_lengths[bs])
# 	return loss

def get_loss(m_hat, r_hat, m, r, batchSize, seq_lengths):
	m=m.unsqueeze(3)#(batchSize,max_seq_lengths,3,1)
	r=r.unsqueeze(3)
	loss=0
	n_mixtures=int(r_hat.size(2)/3)
	m_hat=m_hat.view(batchSize,-1,3,3*n_mixtures)
	r_hat=r_hat.view(batchSize,-1,1,3*n_mixtures)
	#----preprocessing
	for bs in range(batchSize):
		m_bs_pi=F.softmax(m_hat[bs,:seq_lengths[bs],:,:n_mixtures],dim=2)
		r_bs_pi=F.softmax(r_hat[bs,:seq_lengths[bs],:,:n_mixtures],dim=2)
		m_bs_mu=m_hat[bs,:seq_lengths[bs],:,n_mixtures:2*n_mixtures]#(seq_lengths_bs,3,n_mixtures)
		r_bs_mu=r_hat[bs,:seq_lengths[bs],:,n_mixtures:2*n_mixtures]
		m_bs_std=torch.exp(torch.clamp(m_hat[bs,:seq_lengths[bs],:,2*n_mixtures:3*n_mixtures],max=10))+1e-10#add small constant to prevent mode collapse
		r_bs_std=torch.exp(torch.clamp(r_hat[bs,:seq_lengths[bs],:,2*n_mixtures:3*n_mixtures],max=10))+1e-10#add small constant to prevent mode collapse
		m_bs_var=torch.mul(m_bs_std,m_bs_std)
		r_bs_var=torch.mul(r_bs_std,r_bs_std)
		#---calculate neg. log-likelihood
		d_m=m_bs_mu-m[bs,:seq_lengths[bs],:,:]#broadcasting
		d_r=r_bs_mu-r[bs,:seq_lengths[bs],:,:]#broadcasting
		exp_m=torch.exp(-torch.div(torch.mul(d_m,d_m),2*m_bs_var))
		exp_r=torch.exp(-torch.div(torch.mul(d_r,d_r),2*r_bs_var))
		modes_m=torch.div(exp_m,math.sqrt(2*math.pi)*m_bs_std)
		modes_r=torch.div(exp_r,math.sqrt(2*math.pi)*r_bs_std)
		distr_m=torch.sum(torch.mul(modes_m,m_bs_pi),2)
		distr_r=torch.sum(torch.mul(modes_r,r_bs_pi),1)
		loss+=(-torch.sum(torch.log(distr_m))-torch.sum(torch.log(distr_r)))/int(seq_lengths[bs])
	return loss

def greedy_ml_sampling(m_hat,r_hat):
	#m_hat and r_hat must have batch_size=1, that is, the shape (1,seq_length,3 or 1 resp.), where the zeropadding upt to max_seq_lengths must be removed
	n_mixtures=int(r_hat.size(2)/3)
	seq_length=int(r_hat.size(1))
	m_hat=m_hat.view(1,-1,3,3*n_mixtures)
	r_hat=r_hat.view(1,-1,1,3*n_mixtures)
	m_mu=m_hat[0,:,:,n_mixtures:2*n_mixtures]#(seq_lengths,3,n_mixtures)
	r_mu=r_hat[0,:,:,n_mixtures:2*n_mixtures]#(seq_lengths,1,n_mixtures)
	m_pi=F.softmax(m_hat[0,:,:,:n_mixtures],dim=2)
	r_pi=F.softmax(r_hat[0,:,:,:n_mixtures],dim=2)
	m_std=torch.exp(m_hat[0,:,:,2*n_mixtures:3*n_mixtures])+1e-10#add small constant to prevent mode collapse
	r_std=torch.exp(r_hat[0,:,:,2*n_mixtures:3*n_mixtures])+1e-10#add small constant to prevent mode collapse
	norm_m=m_pi/m_std
	norm_r=r_pi/r_std
	amax_idx_m=norm_m.max(2, keepdim=True)[1]
	amax_idx_r=norm_r.max(2, keepdim=True)[1]
	return torch.gather(m_mu,2,amax_idx_m), torch.gather(r_mu,2,amax_idx_r)

def mode_sampling(m_hat,r_hat):
	#m_hat and r_hat must have batch_size=1, that is, the shape (1,seq_length,3 or 1 resp.), where the zeropadding upt to max_seq_lengths must be removed
	n_mixtures=int(r_hat.size(2)/3)
	seq_length=int(r_hat.size(1))
	m_hat=m_hat.view(1,-1,3,3*n_mixtures)
	r_hat=r_hat.view(1,-1,1,3*n_mixtures)
	m_mu=m_hat[0,:,:,n_mixtures:2*n_mixtures]#(seq_lengths,3,n_mixtures)
	r_mu=r_hat[0,:,:,n_mixtures:2*n_mixtures]#(seq_lengths,1,n_mixtures)
	m_pi=F.softmax(m_hat[0,:,:,:n_mixtures],dim=2)
	r_pi=F.softmax(r_hat[0,:,:,:n_mixtures],dim=2)
	m_std=torch.exp(m_hat[0,:,:,2*n_mixtures:3*n_mixtures])+1e-10#add small constant to prevent mode collapse
	r_std=torch.exp(r_hat[0,:,:,2*n_mixtures:3*n_mixtures])+1e-10#add small constant to prevent mode collapse
	norm_m=m_pi/m_std
	norm_r=r_pi/r_std
	s_m=torch.sum(norm_m,2,keepdim=True)
	s_r=torch.sum(norm_r,2,keepdim=True)
	norm_pi_m=norm_m/s_m 
	norm_pi_r=norm_r/s_r
	idx_m=torch.zeros(seq_length,3,1).long()
	idx_r=torch.zeros(seq_length,1,1).long()
	for i in range(seq_length):
		idx_m[i,0,0]=torch.from_numpy(np.random.choice(n_mixtures, 1, p=norm_pi_m[i,0,:].numpy()))
		idx_m[i,1,0]=torch.from_numpy(np.random.choice(n_mixtures, 1, p=norm_pi_m[i,1,:].numpy()))
		idx_m[i,2,0]=torch.from_numpy(np.random.choice(n_mixtures, 1, p=norm_pi_m[i,2,:].numpy()))
		idx_r[i,0,0]=torch.from_numpy(np.random.choice(n_mixtures, 1, p=norm_pi_r[i,0,:].numpy()))
	return torch.gather(m_mu,2,idx_m), torch.gather(r_mu,2,idx_r)



def get_reg_loss(m_hat,r_hat,m,r):
	L=m_hat.size(0)
	d_m=m_hat-m
	d_r=r_hat-r
	loss=(torch.sum(torch.mul(d_m,d_m))+torch.sum(torch.mul(d_r,d_r)))/L
	return loss

def train_environment_model(environment_model,optimizer,data,n_epochs=1000,save_path='dream_models/environment_model.pkl',print_every=200,stop_loss=0.0001):
	total_batch_size=len(data)
	test_data=data[2*int(total_batch_size/3):]
	train_data=data[0:2*int(total_batch_size)]
	train_batch_size=len(train_data)
	test_batch_size=len(test_data)
	# print(train_data[0][0])
	for i in range(n_epochs):
		optimizer.zero_grad()
		packed_sequences=pack_sequences(train_data)
		out_m, out_r, h_rare=environment_model(packed_sequences,train_batch_size)
		unpacked_sequences,seq_lengths=unpack_sequences(packed_sequences)
		m=unpacked_sequences[:,:,0:3]
		r=unpacked_sequences[:,:,3].unsqueeze(2)
		loss=get_loss(out_m,out_r,m,r,train_batch_size,seq_lengths)
		loss.backward()
		optimizer.step()
		if i%print_every==0:
			print('train loss: '+str(i)+' : '+str(loss.item()))
			with torch.no_grad():
				packed_sequences=pack_sequences(test_data)
				out_m, out_r, h_rare=environment_model(packed_sequences,test_batch_size)
				unpacked_sequences,seq_lengths=unpack_sequences(packed_sequences)
				m=unpacked_sequences[:,:,0:3]
				r=unpacked_sequences[:,:,3].unsqueeze(2)
				test_loss=get_loss(out_m,out_r,m,r,test_batch_size,seq_lengths)
				print('test loss: '+str(i)+' : '+str(test_loss.item()))

		if loss.item()<stop_loss:
			print('loss smaller than stop_loss -> stop training')
			torch.save({'model_state': environment_model.state_dict(),'optimizer_state': optimizer.state_dict()}, save_path)
			return environment_model
	torch.save({'model_state': environment_model.state_dict(),'optimizer_state': optimizer.state_dict()}, save_path)
	return environment_model, optimizer

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

def transform_data_to_alpha_and_norm(data):
	data_point=np.zeros(6)
	for i in range(len(data)):
		data_point[:-2]=data[i][0][:-2]
		alpha=0
		ds_norm=0
		data_point[-2]=ds_norm
		data_point[-1]=alpha
		data[i][0]=np.copy(data_point)
		orientation=np.array([1,0])
		for j in range(1,len(data[i])):
			data_point[:-2]=data[i][j][:-2]
			ds_norm=np.linalg.norm(data[i][j][4:])
			if ds_norm==0:
				alpha=0
			else:
				alpha=angle_between(data[i][j][4:], orientation)
				if abs(angle_between(rotation(alpha,orientation),data[i][j][4:]))>0.0001:
					alpha=-alpha
				orientation=data[i][j][4:].copy()
			data_point[-2]=ds_norm
			data_point[-1]=alpha
			data[i][j]=np.copy(data_point)
	return data

def plot_car_perspective(data_one_car,game,car):
	#data must be the data for only one car!
	l=len(data_one_car)
	car_data=np.zeros((l,6))
	for i in range(l):
		car_data[i,:]=data_one_car[i]
	game.plot_halucination(car_data[:90,4:],car_data[:90,0:3],car_data[:90,3],imsize=110,path='gifs/car_perspective.gif',car=car,car_shape=True)


def put_on_car(map_im,position,orientation,backward,path='cars/car_1.png',size_car=16,winner_car=False,car_shape=None):
	position=position.astype(int)
	winner_weight=0.7
	car_im = Image.open(path).convert('RGB')
	map_im=map_im.convert('RGB')
	phi=np.arctan(orientation[0]/(orientation[1]+1e-10))*360/(2*np.pi)+180#need to add 180 because show() has y-coordinate southwards
	if orientation[1]<0:
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
	#rotate(phi+180, expand=True)
	w,h=car_im.size
	W,H=map_im.size
	w_2=int(w/2)
	h_2=int(h/2)
	car_im_px=car_im.load()
	map_im_px=map_im.load()
	winner_idx=np.zeros(4)
	for i in range(w):
		for j in range(h):
			sp=car_im_px[i,j][0]+car_im_px[i,j][1]+car_im_px[i,j][2]
			# idx_w=int(position[0])-w_2+i
			idx_w=position[0]-w_2+i
			idx_w=int(min(max(idx_w,0),W-1))
			# idx_h=int(position[1])-h_2+j
			idx_h=position[1]-h_2+j
			idx_h=int(min(max(idx_h,0),H-1))
			if winner_car:
				if i==0 or i==w-1 or j==0 or j==h-1:
					map_im_px[idx_w,idx_h]=(255,215,0)
			if 10<=sp<700 and 0<=idx_w<W and 0<=idx_h<H:
				if winner_car:
					map_im_px[idx_w,idx_h]=tuple((winner_weight*np.asarray(car_im_px[i,j])+(1-winner_weight)*np.array([255,215,0])).astype(int))
				else:
					map_im_px[idx_w,idx_h]=car_im_px[i,j]
	return map_im

def pil_list_to_cv2(pil_list):
	#converts a list of pil images to a list of cv2 images
	png_list=[]
	for pil_img in pil_list:
		pil_img.save('trash_image.png',format='png')
		png_list.append(cv2.imread('trash_image.png'))
	os.remove('trash_image.png')
	return png_list


# Video Generating function 
def generate_video(cv2_list,path='car_race.avi',fps=10): 
	#makes a video from a given cv2 image list
	if len(cv2_list)==0:
		raise ValueError('the given png list is empty!')
		# image_folder =  '.'# make sure to use your folder 
	video_name = path
	# os.chdir("C:\\Python\\Geekfolder2") 
	# images = [img for img in os.listdir(image_folder) 
	#           if img.endswith(".jpg") or
	#              img.endswith(".jpeg") or
	#              img.endswith("png")] 
	# Array images should only consider 
	# the image files ignoring others if any 
	# print(images)  
	# frame = cv2.imread(os.path.join(image_folder, images[0]))
	frame=cv2_list[0] 
	# setting the frame width, height width 
	# the width, height of first image 
	height, width, layers = frame.shape   
	video = cv2.VideoWriter(video_name, 0, fps, (width, height))  
	# Appending the images to the video one by one 
	for cv2_image in cv2_list:  
	    # video.write(cv2.imread(os.path.join(image_folder, image)))  
	    video.write(cv2_image) 
	# Deallocating memories taken for window creation 
	cv2.destroyAllWindows()  
	video.release()  # releasing the video generated 
  
  

