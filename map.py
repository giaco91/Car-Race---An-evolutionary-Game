import sys
from copy import deepcopy

import numpy as np
from PIL import Image,ExifTags,ImageFilter,ImageOps, ImageDraw
import PIL
import numpy as np
import pickle

import matplotlib.pyplot as plt

from utils import *


#----MAP-------
class Map():
    def __init__(self,size=4):
    	self.size=size
    	if self.size>=5:
    		self.min_map_length=2*self.size
    	else:
    		self.min_map_length=(self.size-1)**2+2
    	skeleton=self.create_map()
    	self.id=()
    	self.skeleton=[np.asarray(skeleton[0])]
    	for i in range(1,len(skeleton)):
    		self.id+=skeleton[i-1]
    		self.skeleton.append(np.asarray(skeleton[i]))
    	# print(self.skeleton[2])
    	# print(self.id[4])
    	# print(self.id[5])
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
    	reflect_y_axis(im).show()

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
    		reflect_y_axis(im).show()
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