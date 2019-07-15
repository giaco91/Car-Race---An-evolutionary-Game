import torch
import torch.nn as nn
import torchvision
import sys
from copy import deepcopy

import numpy as np
from PIL import Image,ExifTags,ImageFilter,ImageOps, ImageDraw
import PIL
import numpy as np
import random
import pickle

import matplotlib.pyplot as plt

random.seed(0)
np.random.seed(7)

class Game():
	def __init__(self,race_map,car_list,n_iter=50,dt=0.01,border_frac=3):
		self.map=race_map
		self.n_checkpoints=len(self.map.skeleton)-1
		self.n_cars=len(car_list)
		self.car_list=car_list
		self.n_iter=n_iter
		self.dt=dt
		self.border_frac=border_frac
		self.positions=[]
		self.orientations=[]
		self.backward=[]
		self.scores=np.zeros(self.n_cars)
		self.max_frames=0
		self.winner_car=None
		for nc in range(self.n_cars):
			self.positions.append([np.array([0.5,0])])
			self.orientations.append([np.array([1,0])])
			self.backward.append([False])

	def max_distance_race(self):
		print('simulate race...')
		self.max_frames=self.n_iter
		for nc in range(self.n_cars):
			c_rot=np.zeros(2)
			smalled_car_size=0.9*self.car_list[nc].size#kosmetik
			crash=False
			for ni in range(self.n_iter):
				#----sensing the environment-----
				inputs=self.get_inputs(nc)

				# inputs=np.array([0,0,0,0,0,0])
				#----decide for action
				a=self.car_list[nc].get_a(inputs)
				if (self.car_list[nc].v>self.car_list[nc].v_max and a>0) or self.car_list[nc].v<-self.car_list[nc].v_max and a<0:
					print('v_max reached: '+str(self.car_list[nc].v))
					a=0
				if crash:
					a=0
				c=self.car_list[nc].get_c(inputs)
				c_rot[0]=-self.orientations[nc][-1][1]
				c_rot[1]=self.orientations[nc][-1][0]

				#--action----
				ds=self.orientations[nc][-1]*self.dt*self.car_list[nc].v
				ds+=c_rot*c*np.linalg.norm(ds)
				ds+=self.orientations[nc][-1]*0.5*a*self.dt**2
				norm_ds=np.linalg.norm(ds)+1e-10

				#---check boundary conditions (crash)
				t_p1,_=self.map.closest_intersection(self.positions[nc][-1],ds,self.map.border1)
				t_p2,_=self.map.closest_intersection(self.positions[nc][-1],ds,self.map.border2)
				t_p=min(t_p1,t_p2)
				if t_p<(norm_ds+smalled_car_size)/norm_ds:
					crash=True
					self.car_list[nc].v=0
					t_crash=(t_p*norm_ds-smalled_car_size)/norm_ds
					self.positions[nc].append(self.positions[nc][-1]+t_crash*ds)
					self.scores[nc]+=t_crash*norm_ds
				#--update the states
				else:
					self.car_list[nc].v+=a*self.dt
					self.positions[nc].append(self.positions[nc][-1]+ds)
					self.scores[nc]+=norm_ds
				if self.car_list[nc].v<0:
					self.backward[nc].append(True)
				elif self.car_list[nc].v>0:
					self.backward[nc].append(False)
				else:
					self.backward[nc].append(self.backward[nc][-1])
				abs_orientation=ds/norm_ds
				if crash:
					self.orientations[nc].append(self.orientations[nc][-1])
				else:
					if self.backward[nc][-1]:
						self.orientations[nc].append(-abs_orientation)
					else:
						self.orientations[nc].append(abs_orientation)
			self.car_list[nc].v=0
			self.winner_car=np.argmax(scores)


	def max_rounds_race(self,c_weight=0.5):
		print('simulate race...')
		for nc in range(self.n_cars):
			last_c=0
			c_rot=np.zeros(2)
			checkpoint_counter=0#first argument is current round, second is the current checkpoint
			delta=0
			inputs=np.zeros(self.car_list[nc].n_inputs)
			smalled_car_size=0.9*self.car_list[nc].size#kosmetik
			crash=False
			for ni in range(self.n_iter):
				if crash:
					self.positions[nc].append(self.positions[nc][-1])
					self.orientations[nc].append(self.orientations[nc][-1])
					self.backward[nc].append(self.backward[nc][-1])
				else:
					#----sensing the environment-----
					inputs=self.get_inputs(nc)
					#----decide for action
					a=self.car_list[nc].get_a(inputs)
					if (self.car_list[nc].v>self.car_list[nc].v_max and a>0) or self.car_list[nc].v<-self.car_list[nc].v_max and a<0:
					# print('v_max reached: '+str(self.car_list[nc].v))
						a=0
					c=c_weight*self.car_list[nc].get_c(inputs)+(1-c_weight)*last_c
					last_c=c
					c_rot[0]=-self.orientations[nc][-1][1]
					c_rot[1]=self.orientations[nc][-1][0]

					#--action----
					ds=self.orientations[nc][-1]*self.dt*self.car_list[nc].v
					ds+=c_rot*c*np.linalg.norm(ds)
					ds+=self.orientations[nc][-1]*0.5*a*self.dt**2
					norm_ds=np.linalg.norm(ds)+1e-10

					#---check boundary conditions (crash)
					t_p1,_=self.map.closest_intersection(self.positions[nc][-1],ds,self.map.border1)
					t_p2,_=self.map.closest_intersection(self.positions[nc][-1],ds,self.map.border2)
					t_p=min(t_p1,t_p2)
					if t_p<(norm_ds+smalled_car_size)/norm_ds:
						crash=True
						if ni+1>self.max_frames:
							self.max_frames=ni+1
						self.car_list[nc].v=0
						t_crash=(t_p*norm_ds-smalled_car_size)/norm_ds
						self.positions[nc].append(self.positions[nc][-1]+t_crash*ds)
					#--update the states
					else:
						self.car_list[nc].v+=a*self.dt
						self.positions[nc].append(self.positions[nc][-1]+ds)
					if self.car_list[nc].v<0:
						self.backward[nc].append(True)
					elif self.car_list[nc].v>0:
						self.backward[nc].append(False)
					else:
						self.backward[nc].append(self.backward[nc][-1])
					abs_orientation=ds/norm_ds
					if crash:
						self.orientations[nc].append(self.orientations[nc][-1])
					else:
						if self.backward[nc][-1]:
							self.orientations[nc].append(-abs_orientation)
						else:
							self.orientations[nc].append(abs_orientation)
					checkpoint,delta=self.get_checkpoint(nc)
					if np.mod(checkpoint_counter+1,self.n_checkpoints)==checkpoint:
						checkpoint_counter+=1
					elif np.mod(checkpoint_counter-1,self.n_checkpoints)==checkpoint:
						checkpoint_counter-=1
			self.scores[nc]+=checkpoint_counter+delta
			self.car_list[nc].v=0
			if self.max_frames==0:
				self.max_frames=self.n_iter
			self.winner_car=np.argmax(self.scores)

	def get_inputs(self,nc):
		inputs=np.zeros(self.car_list[nc].n_inputs)
		s0=self.orientations[nc][-1]
		s1=rotation(np.pi/4,self.orientations[nc][-1])
		s2=rotation(-np.pi/4,self.orientations[nc][-1])
		t_p01,t_n01=self.map.closest_intersection(self.positions[nc][-1],s0,self.map.border1)
		t_p02,t_n02=self.map.closest_intersection(self.positions[nc][-1],s0,self.map.border2)
		t_p11,t_n11=self.map.closest_intersection(self.positions[nc][-1],s1,self.map.border1)
		t_p12,t_n12=self.map.closest_intersection(self.positions[nc][-1],s1,self.map.border2)
		t_p21,t_n21=self.map.closest_intersection(self.positions[nc][-1],s2,self.map.border1)
		t_p22,t_n22=self.map.closest_intersection(self.positions[nc][-1],s2,self.map.border2)
		inputs[0]=min(t_p01,t_p02)
		inputs[1]=max(t_n01,t_n02)
		inputs[2]=min(t_p11,t_p12)
		inputs[3]=max(t_n11,t_n12)
		inputs[4]=min(t_p21,t_p22)
		inputs[5]=max(t_n21,t_n22)
		return inputs

	def get_checkpoint(self,nc):
		d=np.zeros(self.n_checkpoints)
		for c in range(self.n_checkpoints):
			d[c]=np.linalg.norm(self.positions[nc][-1]-self.map.skeleton[c])
		sorted_idx=np.argsort(d)
		if sorted_idx[0]==np.mod(sorted_idx[1]+1,self.n_checkpoints):
			delta=-d[sorted_idx[0]]
		else:
			delta=d[sorted_idx[0]]
		return sorted_idx[0],delta

				
	def plot_game(self,path='car_race.gif',imsize=256):
		print('rendering ...')
		frames=[]
		_,_,map_im=self.map.draw_map(imsize=imsize,show=False,border_frac=self.border_frac)
		scale=imsize/(self.map.size-1)
		border=int(scale/self.border_frac)
		for ni in range(self.max_frames):
			pil_f=map_im.copy()
			for nc in range(self.n_cars):
				sc=scale*self.positions[nc][ni]+border
				if self.winner_car==nc:
					winner_car=True
				else:
					winner_car=False
				pil_f=self.put_on_car(pil_f,(max(0,sc[0]),max(sc[1],0)),self.orientations[nc][ni],self.backward[nc][ni],size_car=int(self.car_list[nc].size*scale),path='cars/car_'+str(self.car_list[nc].model)+'.png',winner_car=winner_car)
			frames.append(reflect_y_axis(pil_f))
		print(len(frames))
		frames[0].save(path,
		               save_all=True,
		               append_images=frames[1:],
		               duration=10*self.dt/0.01,
		               loop=0)

	def put_on_car(self,map_im,position,orientation,backward,path='cars/car_2.png',size_car=16,winner_car=False):
		winner_weight=0.8
		car_im = Image.open(path).convert('RGB')
		map_im=map_im.convert('RGB')
		phi=np.arctan(orientation[0]/(orientation[1]+1e-10))*360/(2*np.pi)+180#need to add 180 because show() has y-coordinate southwards
		if orientation[1]<0:
			phi+=180
		car_im=resize_to_height_ref(car_im,size_car).rotate(phi, expand=True)
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
				idx_w=int(position[0])-w_2+i
				idx_h=int(position[1])-h_2+j
				if winner_car:
					if i==0 or i==w-1 or j==0 or j==h-1:
						map_im_px[idx_w,idx_h]=(255,200,0)
				if 10<sp<750 and 0<=idx_w<W and 0<=idx_h<H:
					if winner_car:
						map_im_px[idx_w,idx_h]=tuple((winner_weight*np.asarray(car_im_px[i,j])+(1-winner_weight)*np.array([255,255,0])).astype(int))
					else:
						map_im_px[idx_w,idx_h]=car_im_px[i,j]
		return map_im

	def selection_and_mutation(self,N_sel,N_mut):
		print('selection and mutation ....')
		sorted_idx=np.argsort(self.scores)
		print('Best score: '+str(self.scores[sorted_idx[-1]]))
		selected_cars=[]
		mutated_cars=[]
		mutation_rate=[]
		for i in range(N_sel):
			selected_cars.append(self.car_list[sorted_idx[-1-i]])
			mutation_rate.append(1+100/(1+self.scores[sorted_idx[-1-i]]))

		with open('best_cars/best_car_map.pkl', 'wb') as f:
			pickle.dump(selected_cars[0], f )	

		if N_sel>=self.n_cars:
			print('Not enough cars! No selection!')
		mutated_cars.append(selected_cars[0])
		n=1
		while n<=N_mut-1:
			mutated_cars.append(deepcopy(selected_cars[np.mod(n,N_sel)]))
			mutated_cars[-1].mutation(mutation_rate=mutation_rate[np.mod(n,N_sel)])
			n+=1
		return mutated_cars


#–---CAR-----
class Car():
	def __init__(self,v_max=5,grip=0,m=1000,F_max=1000,size=0.1,model=2,backward=0.1):
		self.model=model
		self.v=0
		self.F_max=F_max
		self.m=m
		self.size=size
		self.a_max=F_max/m
		self.v_max=v_max
		self.n_inputs=6
		self.n_h=3
		self.grip=grip#smaller or equal zero. zero is perfect grip. -1 is not good grip
		self.backward=backward#the capability to drive backward compared to forward
		self.a_weights=np.random.rand(self.n_h+1)-0.5
		self.c_weights=(np.random.rand(self.n_h+1)-0.5)
		self.h_weights=np.random.rand(self.n_inputs+2,self.n_h)-0.5#parametersharing of hidden representation

	def mutation(self,mutation_rate=1):
		#mutation_rate can be any positive number
		self.a_weights+=(np.random.rand(self.n_h+1)-0.5)*0.1*mutation_rate
		self.c_weights+=(np.random.rand(self.n_h+1)-0.5)*0.1*mutation_rate
		self.h_weights+=(np.random.rand(self.n_inputs+2,self.n_h)-0.5)*0.1*mutation_rate
		# print(np.linalg.norm(self.a_weights))
		# print(np.linalg.norm(self.c_weights))

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


#----MAP-------
class Map():
    def __init__(self,size=4):
    	self.size=size
    	if self.size>=5:
    		self.min_map_length=2*self.size
    	else:
    		self.min_map_length=(self.size-1)**2+2
    	skeleton=self.create_map()
    	for i in range(len(skeleton)):
    		skeleton[i]=np.asarray(skeleton[i])
    	self.skeleton=skeleton
    	border1,border2,map_im=self.draw_map(show=False)
    	for i in range(len(border1)):
    		border1[i]=np.asarray(border1[i])
    		border2[i]=np.asarray(border2[i])
    	self.border1=border1
    	self.border2=border2
    	self.map_im=map_im

    def create_map(self):
    	trace=[]#coordinates of skeleton
    	trace.append((0,0))
    	trace.append((1,0))
    	direction=int(np.random.choice(2, 1))
    	if direction==0:
    		trace.append((2,0))
    	else:
    		trace.append((1,1))
    	return self.recurrent_map_creation(trace,trace.copy())


    def recurrent_map_creation(self,trace,initial_trace):
    	xy_now=trace[-1]
    	xy_last=trace[-2]
    	allowed_directions=[]
    	possible_next_coordinate=[]
    	possible_next_coordinate.append((xy_now[0]+1,xy_now[1]))
    	if possible_next_coordinate[-1][0]<=self.size-1 and  possible_next_coordinate[-1] not in trace[1:-1]:
    		allowed_directions.append(0)
    	possible_next_coordinate.append((xy_now[0],xy_now[1]+1))
    	if possible_next_coordinate[-1][1]<=self.size-1 and  possible_next_coordinate[-1] not in trace[1:-1]:
    		allowed_directions.append(1)
    	possible_next_coordinate.append((xy_now[0]-1,xy_now[1]))
    	if possible_next_coordinate[-1][0]>=0 and  possible_next_coordinate[-1] not in trace[1:-1]:
    		allowed_directions.append(2)
    	possible_next_coordinate.append((xy_now[0],xy_now[1]-1))
    	if possible_next_coordinate[-1][1]>=0 and  possible_next_coordinate[-1] not in trace[1:-1]:
    		allowed_directions.append(3)
    	
    	if len(allowed_directions)>=1:
    		direction=int(np.random.choice(allowed_directions, 1))#0=right,1=up,2=left,3=bot
    		trace.append(possible_next_coordinate[direction])
    	else: 
    		return self.recurrent_map_creation(initial_trace,initial_trace.copy())
    	if trace[-1]==trace[0] and len(trace)>=self.min_map_length:
    		return trace
    	else:
    		return self.recurrent_map_creation(trace,initial_trace)

    def closest_intersection(self,p,d,segment_list):
    	positive_s_points=[]
    	negative_s_points=[]
    	t_pos=1e10
    	t_neg=-1e10
    	for i in range(len(segment_list)-1):
    		t_i=intersection_of_two_lines(p,d,segment_list[i],segment_list[i+1]-segment_list[i])
    		if t_i[1] is not None:
	    		if 0<=t_i[1]<=1:
	    			if t_i[0]>=0:
	    				positive_s_points.append(t_i[0])
	    			else:
	    				negative_s_points.append(t_i[0])
    	if len(positive_s_points)>0:
    		t_pos=np.min(np.asarray(positive_s_points))
    	if len(negative_s_points)>0:
    		t_neg=np.max(np.asarray(negative_s_points))
    	return t_pos,t_neg

    def draw_skeleton(self,skeleton=None,imsize=256):
    	#skeleton is an (ordered) list of coordinates
    	if skeleton is None:
    		skeleton=self.skeleton
    	scale=imsize/(self.size-1)
    	border=int(scale/3)
    	width=int(imsize/100)
    	im=create_image(imsize+2*border,imsize+2*border)
    	draw = ImageDraw.Draw(im)
    	coordinates=[tuple(scale*c+border for c in t) for t in skeleton]
    	draw.line(coordinates,fill=(0,0,255),width=width)
    	im.show()

    def draw_map(self,skeleton=None,imsize=256,show=True,border_frac=3):
    	if skeleton is None:
    		skeleton=self.skeleton
    	scale=imsize/(self.size-1)
    	border=int(scale/border_frac)
    	width=int(imsize/100)
    	im=create_image(imsize+2*border,imsize+2*border)
    	W,H=im.size
    	px_im=im.load()
    	for w in range(W):
    		for h in range(H):
    			px_im[w,h]=(0,200,0)
    	draw = ImageDraw.Draw(im)
    	cb1,cb2=self.get_border_coordinates(skeleton=skeleton)
    	coordinates=[tuple(scale*c+border for c in t) for t in skeleton]
    	ccb1=[tuple(scale*c+border for c in t) for t in cb1]
    	ccb2=[tuple(scale*c+border for c in t) for t in cb2]
    	draw.line(coordinates,fill=(190,190,190),width=int(2*border))
    	draw.line(ccb1,fill=(200,0,50),width=width)
    	draw.line(ccb2,fill=(200,0,50),width=width)
    	if show:
    		im.show()
    	return cb1,cb2,im



    def get_border_coordinates(self,skeleton=None):
    	if skeleton is None:
    		skeleton=self.skeleton
    	cb1=[]
    	cb2=[]
    	d=0.25
    	cb1.append((-d,-d))
    	cb2.append((d,d))
    	s=0#state of direction: 0=east,1=north,2=west,3=south
    	for i in range(2,len(skeleton)):
    		if tuple(skeleton[i])==(skeleton[i-1][0]+1,skeleton[i-1][1]):
    			s_next=0
    		elif tuple(skeleton[i])==(skeleton[i-1][0],skeleton[i-1][1]+1):
    			s_next=1
    		elif tuple(skeleton[i])==(skeleton[i-1][0]-1,skeleton[i-1][1]):
    			s_next=2
    		elif tuple(skeleton[i])==(skeleton[i-1][0],skeleton[i-1][1]-1):
    			s_next=3
    		else:
    			print(skeleton[i-1])
    			print(skeleton[i])
    			raise ValueError('inconsistency detected in skeleton')

    		if s==0 or s==2:
    			dc1y=d
    			dc1x=0
    			if s==0:
    				dc1y=-d
    			if s_next==1:
    				dc1x=d
    			elif s_next==3:
    				dc1x=-d

    		if s==1 or s==3:
    			dc1x=-d
    			dc1y=0
    			if s==1:
    				dc1x=d
    			if s_next==2:
    				dc1y=d
    			elif s_next==0:
    				dc1y=-d
    		cb1.append((skeleton[i-1][0]+dc1x,skeleton[i-1][1]+dc1y))
    		cb2.append((skeleton[i-1][0]-dc1x,skeleton[i-1][1]-dc1y))
    		s=s_next
    	cb1.append(cb1[0])
    	cb2.append(cb2[0])
    	return cb1,cb2




def create_image(i, j):
	image = Image.new("RGB", (i, j), "white")
	return image

def sigmoid(s,bias=0):
	return 1/(1+np.exp(-s))+bias

def resize_to_height_ref(image,n_height):
    w,h=image.size
    return image.resize((n_height,round(n_height*h/w)),Image.ANTIALIAS)

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


