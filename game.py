import sys
from copy import deepcopy

import numpy as np
from PIL import Image,ExifTags,ImageFilter,ImageOps, ImageDraw
import PIL
import numpy as np
import pickle

import matplotlib.pyplot as plt

from utils import *



class Game():
	def __init__(self,race_map,car_list,n_iter=50,dt=0.01,border_frac=3,save_path='best_cars/best_car.pkl'):
		self.save_path=save_path
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
				t_p1,_=self.map.closest_intersection(self.positions[nc][-1],ds,self.map.border1,get_also_directions=True)
				t_p2,_=self.map.closest_intersection(self.positions[nc][-1],ds,self.map.border2,get_also_directions=True)
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

	def max_rounds_race_only_front_sight(self,c_weight=0.8):
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
					a=self.car_list[nc].get_a(inputs[0::2])
					if (self.car_list[nc].v>self.car_list[nc].v_max and a>0) or self.car_list[nc].v<-self.car_list[nc].v_max and a<0:
					# print('v_max reached: '+str(self.car_list[nc].v))
						a=0
					c=c_weight*self.car_list[nc].get_c(inputs[0::2])+(1-c_weight)*last_c
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

	def max_rounds_race_shape(self,c_weight=0.8,get_data=False):
		print('simulate race...')
		if get_data:
			data=[]#list of sequences, where each sequence point contains the information (measurement/inputs,d_score,action)
		for nc in range(self.n_cars):
			if get_data:
				data.append([])
				score_tracker=self.positions[nc][0][0]
			self.car_list[nc].transform_shape()
			last_c=0
			c_rot=np.zeros(2)
			checkpoint_counter=0#first argument is current round, second is the current checkpoint
			delta=0
			inputs=np.zeros(self.car_list[nc].n_inputs)
			longitudinal_car_size=0.5*self.car_list[nc].size*self.car_list[nc].aerodynamic
			crash=False
			for ni in range(self.n_iter):
				if get_data:
					data_point=np.zeros(6)#3 measurment inputs, 1 score, 2 ds
				if crash:
					self.positions[nc].append(self.positions[nc][-1])
					self.orientations[nc].append(self.orientations[nc][-1])
					self.backward[nc].append(self.backward[nc][-1])
				else:
					#----sensing the environment-----
					inputs=self.get_inputs(nc)
					if get_data:
						data_point[0:3]=inputs[0::2]
					#----decide for action
					a=self.car_list[nc].get_a(inputs[0::2])
					if (self.car_list[nc].v>self.car_list[nc].v_max and a>0) or self.car_list[nc].v<-self.car_list[nc].v_max and a<0:
					# print('v_max reached: '+str(self.car_list[nc].v))
						a=0
					c=c_weight*self.car_list[nc].get_c(inputs[0::2])+(1-c_weight)*last_c
					last_c=c
					c_rot[0]=-self.orientations[nc][-1][1]
					c_rot[1]=self.orientations[nc][-1][0]

					#--action----
					ds=self.orientations[nc][-1]*self.dt*self.car_list[nc].v
					ds+=c_rot*c*np.linalg.norm(ds)
					ds+=self.orientations[nc][-1]*0.5*a*self.dt**2
					norm_ds=np.linalg.norm(ds)+1e-10

					#---check boundary conditions (crash)
					t_p1,_,d_p1,_=self.map.closest_intersection(self.positions[nc][-1],ds,self.map.border1,get_also_directions=True)
					t_p2,_,d_p2,_=self.map.closest_intersection(self.positions[nc][-1],ds,self.map.border2,get_also_directions=True)
					t_list=[t_p1,t_p2]
					d_list=[d_p1,d_p2]
					closes_idx=np.argmin(t_list)
					t_p=t_list[closes_idx]
					d_p=d_list[closes_idx]
					if tuple(d_p)==tuple(ds):
						do=0
						alpha=0
					else:
						alpha=get_angle(d_p,ds)
						do=abs((self.car_list[nc].size/2)/np.tan(alpha))
						if alpha==0:
							raise ValueError('alpha is zero')
					if t_p<(norm_ds+longitudinal_car_size+do)/norm_ds:
						crash=True
						if ni+1>self.max_frames:
							self.max_frames=ni+1
						self.car_list[nc].v=0
						t_crash=max(0,(t_p*norm_ds-(longitudinal_car_size+do))/norm_ds)
						self.positions[nc].append(self.positions[nc][-1]+t_crash*ds)

					#--update the states
					else:
						self.car_list[nc].v+=a*self.dt
						self.positions[nc].append(self.positions[nc][-1]+ds)
						if get_data:
							data_point[4:]=ds
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
					if get_data:
						data_point[3]=checkpoint_counter+delta-score_tracker
						score_tracker=checkpoint_counter+delta
					data[nc].append(data_point)
			self.scores[nc]+=checkpoint_counter+delta
			self.car_list[nc].v=0
			if self.max_frames==0:
				self.max_frames=self.n_iter
			self.winner_car=np.argmax(self.scores)
		if get_data:
			return data


	def get_inputs(self,nc):
		inputs=np.zeros(6)
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

				
	def plot_game(self,path='car_race.gif',imsize=256,car_shape=False):
		print('rendering ...')
		frames=[]
		_,_,map_im=self.map.draw_map(imsize=imsize,show=False,border_frac=self.border_frac)

		scale=imsize/(self.map.size-1)
		border=int(scale/self.border_frac)
		for ni in range(self.max_frames):
			pil_f=map_im.copy()
			for nc in range(self.n_cars):
				if car_shape:
					car_shape=self.car_list[nc].aerodynamic
				else:
					car_shape=None
				sc=scale*self.positions[nc][ni]+border
				if self.winner_car==nc:
					winner_car=True
				else:
					winner_car=False
				pil_f=self.put_on_car(pil_f,(max(0,sc[0]),max(sc[1],0)),self.orientations[nc][ni],self.backward[nc][ni],size_car=int(max(1,self.car_list[nc].size*scale)),path='cars/car_'+str(self.car_list[nc].model)+'.png',winner_car=winner_car,car_shape=car_shape)
			frames.append(reflect_y_axis(pil_f))
		print(len(frames))
		frames[0].save(path,
		               save_all=True,
		               append_images=frames[1:],
		               duration=10*self.dt/0.01,
		               loop=0)

	def put_on_car(self,map_im,position,orientation,backward,path='cars/car_2.png',size_car=16,winner_car=False,car_shape=False):
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
				idx_w=int(position[0])-w_2+i
				idx_w=int(min(max(idx_w,0),W-1))
				idx_h=int(position[1])-h_2+j
				idx_h=int(min(max(idx_h,0),H-1))
				if winner_car:
					if i==0 or i==w-1 or j==0 or j==h-1:
						map_im_px[idx_w,idx_h]=(255,215,0)
				if 10<sp<750 and 0<=idx_w<W and 0<=idx_h<H:
					if winner_car:
						map_im_px[idx_w,idx_h]=tuple((winner_weight*np.asarray(car_im_px[i,j])+(1-winner_weight)*np.array([255,215,0])).astype(int))
					else:
						map_im_px[idx_w,idx_h]=car_im_px[i,j]
		return map_im

	def selection_and_mutation(self,N_sel,N_mut,mut_fac=1,shape_mutation=False):
		print('selection and mutation ....')
		sorted_idx=np.argsort(self.scores)
		print('Best score: '+str(self.scores[sorted_idx[-1]]))
		selected_cars=[]
		mutated_cars=[]
		mutation_rate=[]
		for i in range(N_sel):
			selected_cars.append(self.car_list[sorted_idx[-1-i]])
			mutation_rate.append(mut_fac+100/(1+self.scores[sorted_idx[-1-i]]))

		print('aerodynamic: '+str(selected_cars[0].aerodynamic))
		print('size: '+str(selected_cars[0].size))
		with open(self.save_path, 'wb') as f:
			pickle.dump(selected_cars[0], f )	

		if N_sel>=self.n_cars:
			print('Not enough cars! No selection!')
		mutated_cars.append(selected_cars[0])
		n=1
		while n<=N_mut-1:
			mutated_cars.append(deepcopy(selected_cars[np.mod(n,N_sel)]))
			mutated_cars[-1].mutation(mutation_rate=mutation_rate[np.mod(n,N_sel)],shape_mutation=shape_mutation)
			n+=1
		return mutated_cars