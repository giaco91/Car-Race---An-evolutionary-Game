import torch
import torch.nn as nn
import torchvision
import sys

import numpy as np
from PIL import Image,ExifTags,ImageFilter,ImageOps, ImageDraw
import PIL
import numpy as np
import random

import matplotlib.pyplot as plt

random.seed(0)
np.random.seed(0)

class Game():
	def __init__(self,race_map,car,n_iter=50,dt=0.01,border_frac=3):
		self.map=race_map
		self.car=car
		self.n_iter=n_iter
		self.positions=[]
		self.positions.append(np.array([0.1,0]))
		self.orientations=[]
		self.orientations.append(np.array([1,0]))
		self.dt=dt
		self.border_frac=border_frac

	def simple_race(self):
		c_rot=np.zeros(2)
		no_crash=True
		i=0
		smalled_car_size=0.9*self.car.size#kosmetik
		while i<self.n_iter and no_crash:
		#for i in range(self.n_iter):
			a=self.car.get_a(np.array([0,0,0,0]))
			if (self.car.v>self.car.v_max and a>0) or self.car.v<-self.car.v_max and a<0:
				#print('v_max reached: '+str(self.car.v))
				a=0
			c=self.car.get_c(np.array([0,0,0,0]))
			c_rot[0]=-self.orientations[-1][1]
			c_rot[1]=self.orientations[-1][0]
			ds=self.orientations[-1]*self.dt*self.car.v
			ds+=c_rot*c*np.linalg.norm(ds)
			ds+=self.orientations[-1]*0.5*a*self.dt**2
			norm_ds=np.linalg.norm(ds)
			t_p1,_=self.map.closest_intersection(self.positions[-1],ds,self.map.border1)
			t_p2,_=self.map.closest_intersection(self.positions[-1],ds,self.map.border2)
			t_p=min(t_p1,t_p2)
			# t_p,_=self.map.closest_intersection(self.positions[-1],ds,self.map.skeleton)
			if t_p<(norm_ds+smalled_car_size)/norm_ds:
				print('crash!')
				#no_crash=False
				self.car.v=0
				t_crash=(t_p*norm_ds-smalled_car_size)/norm_ds
				self.positions.append(self.positions[-1]+t_crash*ds)
			else:
				self.car.v+=a*self.dt
				self.positions.append(self.positions[-1]+ds)
			self.orientations.append(ds/norm_ds)
			i+=1

	def plot_game(self,path='car_race.gif',imsize=256):
		frames=[]
		_,_,map_im=self.map.draw_map(imsize=imsize,show=False,border_frac=self.border_frac)
		# self.put_on_car(map_im,np.array([150,150]))
		scale=imsize/self.map.size
		border=int(scale/self.border_frac)
		for i in range(len(self.positions)):
			sc=scale*self.positions[i]+border
			pil_f=self.put_on_car(map_im.copy(),(max(0,sc[0]),max(sc[1],0)),self.orientations[i],size_car=int(self.car.size*scale))
			frames.append(pil_f)
		print(len(frames))
		frames[0].save(path,
		               save_all=True,
		               append_images=frames[1:],
		               duration=100,
		               loop=0)

	def put_on_car(self,map_im,position,orientation,path='cars/car_2.png',size_car=16):
		car_im = Image.open(path)
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
		for i in range(w):
			for j in range(h):
				sp=car_im_px[i,j][0]+car_im_px[i,j][1]+car_im_px[i,j][2]
				idx_w=int(position[0])-w_2+i
				idx_h=int(position[1])-h_2+j
				if 10<sp<750 and 0<=idx_w<W and 0<=idx_h<H:
					map_im_px[idx_w,idx_h]=car_im_px[i,j]
		return reflect_y_axis(map_im)



class Car():
	def __init__(self,v_max=5,grip=0,m=1000,F_max=10000,size=0.1):
		self.v=0
		self.F_max=F_max
		self.m=m
		self.size=size
		self.a_max=F_max/m
		self.v_max=v_max
		self.n_inputs=6
		self.grip=grip#larger than zero, zero is maximum grip...
		self.a_weights=np.random.rand(self.n_inputs+1)-0.5
		self.c_weights=np.random.rand(self.n_inputs+1)-0.5

	def mutation(self,weights,scale=0.1):
		return weights+(np.random.rand(self.n_inputs)-0.5)*scale

	def get_a(self,inputs):
		s=self.a_weights[0]
		s+=self.a_weights[1]*self.v
		for i in range(inputs.shape[0]):
			s+=self.a_weights[i+2]*inputs[i]
		return sigmoid(s)*self.a_max

	def get_c(self,inputs):
		s=self.c_weights[0]
		s+=self.c_weights[1]*self.v
		for i in range(inputs.shape[0]):
			s+=self.c_weights[i+2]*inputs[i]
		return sigmoid(s)*(abs(self.v)**self.grip)

class Map():
    def __init__(self,size=4):
    	self.size=size
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
    	if trace[-1]==trace[0]:
    		return trace
    	else:
    		return self.recurrent_map_creation(trace,initial_trace)

    def closest_intersection(self,p,d,segment_list):
    	positive_s_points=[]
    	negative_s_points=[]
    	t_pos=1e10
    	t_neg=1e10
    	for i in range(len(segment_list)-1):
    		t_i=intersection_of_two_lines(p,d,segment_list[i],segment_list[i+1]-segment_list[i])
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
    	scale=imsize/self.size
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
    	scale=imsize/self.size
    	border=int(scale/border_frac)
    	width=int(imsize/100)
    	im=create_image(imsize+2*border,imsize+2*border)
    	draw = ImageDraw.Draw(im)
    	cb1,cb2=self.get_border_coordinates(skeleton=skeleton)
    	coordinates=[tuple(scale*c+border for c in t) for t in skeleton]
    	ccb1=[tuple(scale*c+border for c in t) for t in cb1]
    	ccb2=[tuple(scale*c+border for c in t) for t in cb2]
    	draw.line(coordinates,fill=(0,0,255),width=width)
    	draw.line(ccb1,fill=(255,0,0),width=width)
    	draw.line(ccb2,fill=(255,0,0),width=width)
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

def sigmoid(s):
	return 1/(1+np.exp(-s))	

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
		return np.array([1e10,1e10])
	else:
		A_inv=np.linalg.inv(A) 
		t=np.dot(A_inv,p2-p1)
		#s=p1+t[0]*d1
		return t


