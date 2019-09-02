from utils import *
from car import *
from game import *
from map import *
from environment_model import *


from copy import deepcopy
import argparse


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--load_path', default='best_cars/', help='path to load car')
parser.add_argument('--save_path', default='best_cars/', help='path to save car')
parser.add_argument('--map_size', type=int, default=3, help='size of the map')
parser.add_argument('--n_sel', type=int, default=1)
parser.add_argument('--n_mut', type=int, default=20)
parser.add_argument('--n_gen', type=int, default=3)
parser.add_argument('--n_iter', type=int, default=100)
parser.add_argument('--plot_every', type=int, default=1)
parser.add_argument('--n_h', type=int, default=2)
parser.add_argument('--dt', type=float, default=0.08)
parser.add_argument('--load_car', type=str2bool, default=False)

np.random.seed(0)
opt = parser.parse_args()
print(opt)


# img=create_image(1600, 900)
# imsize=700
# draw=ImageDraw.Draw(img)
# dcirc=0.03*imsize
# x_coord=0.9*imsize
# y_coord=0.5*imsize
# for k in range(3):
# 	for l in range(2):
# 		draw.line([(int(x_coord),int(y_coord+3*k*dcirc)),(int(x_coord+0.2*imsize),int(y_coord+3*l*dcirc))],fill=(0,0,0),width=1)
# for l in range(2):
# 	for m in range(2):
# 		draw.line([(int(x_coord+0.2*imsize),int(y_coord+3*l*dcirc)),(int(x_coord+0.4*imsize),int(y_coord+3*m*dcirc))],fill=(0,0,0),width=1)
# for k in range(3):
# 	color=(255,100,100)
# 	size_fac=1
# 	draw.ellipse([(int(x_coord-size_fac*dcirc),int(y_coord-size_fac*dcirc+3*k*dcirc)),(int(x_coord+size_fac*dcirc),int(y_coord+size_fac*dcirc+3*k*dcirc))], fill=color, outline=(0,0,0))
# for l in range(2):
# 	size_fac=1
# 	color=(255,100,100)
# 	draw.ellipse([(int(x_coord+0.2*imsize-size_fac*dcirc),int(y_coord-size_fac*dcirc+3*l*dcirc)),(int(x_coord+0.2*imsize+size_fac*dcirc),int(y_coord+size_fac*dcirc+3*l*dcirc))], fill=color, outline=(0,0,0))
# for m in range(2):
# 	if m==0:
# 		text='force'
# 	else:
# 		text='curve'
# 	color=(255,100,100)
# 	size_fac=1
# 	draw.ellipse([(int(x_coord+0.4*imsize-size_fac*dcirc),int(y_coord-size_fac*dcirc+3*m*dcirc)),(int(x_coord+0.4*imsize+size_fac*dcirc),int(y_coord+size_fac*dcirc+3*m*dcirc))], fill=color, outline=(0,0,0))
# 	# font = ImageFont.truetype("arial.ttf", int(70*imsize/1000))
# 	# draw.text((int(0.5*imsize+3*dcirc),int(0.3*imsize+3*m*dcirc-1.2*dcirc)), text, font=font, fill=(0,0,0))


# img.save('nn.png',format='png')
# raise ValueError('asdf')

load_car=opt.load_car
cars=[]
N_sel=opt.n_sel
N_mut=opt.n_mut
N_gen=opt.n_gen
n_cars=N_mut
plot_every=opt.plot_every

if opt.save_path=='best_cars/':
	save_path=opt.save_path+'nh='+str(opt.n_h)+'.pkl'
else:
	save_path=opt.save_path
if opt.load_path=='best_cars/':
	load_path=opt.load_path+'nh='+str(opt.n_h)+'.pkl'
else:
	load_path=opt.load_path
	print('load car from: '+str(opt.load_path))
# race_map=Map(size=4)
# race_map.draw_map()

# race_map_list=[Map(size=3),Map(size=4),Map(size=5),Map(size=6),Map(size=7),Map(size=8)]
# race_map_list=[Map(size=4),Map(size=4),Map(size=4),Map(size=4),Map(size=4)]
race_map_list=[]
n_race_maps=20
size=opt.map_size
if size>=5:
	min_points=20
else:
	min_points=0
k=0


while len(race_map_list)<=n_race_maps-1 and k<100:
	race_map=Map(size=size)
	different=True
	if len(race_map.skeleton)>=min_points:
		for i in range(len(race_map_list)):
			if race_map_list[i].id==race_map.id:
				different=False
		if different:
			race_map_list.append(race_map)
	k+=1



for j in range(len(race_map_list)):
	# race_map_list[j].draw_map()
	# race_map_list[j].draw_skeleton(imsize=500,save_path='video_figures/skelets/skelet_s='+str(size)+'_'+str(j)+'.png',show=False)
	# race_map_list[j].draw_map(imsize=500,save_path='video_figures/colored/map_s='+str(size)+'_'+str(j)+'.png',show=False)
	# race_map_list[j].draw_map(imsize=500,show=True)
	pass



# race_map_list[-13].draw_map(imsize=500,show=True)
# raise ValueError('asdf')
# race_map_list[0]=race_map_list[-13]


#load
if load_car:
	print('loading car ...')
	with open(load_path, 'rb') as f:
	    best_car = pickle.load(f)
	    cars.append(deepcopy(best_car))
	    for nc in range(n_cars-1):
	    	cars.append(deepcopy(best_car))
	    	cars[-1].mutation(shape_mutation=False)
else:
	for nc in range(0):
		cars.append(Car(size=0.15,model=1,grip=-0.,v_max=1000,n_inputs=3,n_h=15))
	for nc in range(n_cars):
		cars.append(Car(grip=-1,model=3,n_h=opt.n_h,n_inputs=3,mutate_physics=False,v_max=2,size=0.15,F_max=2000))


def get_inputs(orientation,position,map):
	inputs=np.zeros(6)
	s0=orientation
	s1=rotation(np.pi/4,orientation)
	s2=rotation(-np.pi/4,orientation)
	t_p01,t_n01=map.closest_intersection(position,s0,map.border1)
	t_p02,t_n02=map.closest_intersection(position,s0,map.border2)
	t_p11,t_n11=map.closest_intersection(position,s1,map.border1)
	t_p12,t_n12=map.closest_intersection(position,s1,map.border2)
	t_p21,t_n21=map.closest_intersection(position,s2,map.border1)
	t_p22,t_n22=map.closest_intersection(position,s2,map.border2)
	inputs[0]=min(t_p01,t_p02)
	inputs[1]=max(t_n01,t_n02)
	inputs[2]=min(t_p11,t_p12)
	inputs[3]=max(t_n11,t_n12)
	inputs[4]=min(t_p21,t_p22)
	inputs[5]=max(t_n21,t_n22)
	return inputs


car_shape=True
#------draw car on map with laser measure
# imsize=500

# border_frac=3
# dcirc=5
# laser_rgb=(230,200,0)
# map=race_map_list[0]

# position_list=[np.array([0.4,0])]
# orientation_list=[np.array([1,0])]
# for j in range(10):
# 	orientation_list.append(rotation(np.pi/20*j,np.array([1,0])))
# 	position_list.append(position_list[-1]+0.1*orientation_list[-1])

# for i in range(1):
# 	position=position_list[i]
# 	orientation=orientation_list[i]
# 	_,_,map_im=map.draw_map(imsize=imsize,show=False,border_frac=border_frac)
# 	scale=imsize/(race_map_list[0].size-1)
# 	border=int(scale/border_frac)
# 	pil_f=map_im.copy()
# 	car=cars[0]
# 	if car_shape:
# 		car_shape=car.aerodynamic
# 	else:
# 		car_shape=None
# 	sc=scale*position+border
# 	winner_car=False
# 	pil_f=put_on_car(pil_f,np.array([max(0,sc[0]),max(sc[1],0)]),orientation,False,size_car=int(max(1,car.size*scale)),path='cars/car_'+str(car.model)+'.png',winner_car=winner_car,car_shape=car_shape)
# 	# reflect_y_axis(pil_f).show()
# 	reflect_y_axis(pil_f).save('video_figures/images/car_on_map.png',format='png')
# 	#-----measure
# 	draw = ImageDraw.Draw(pil_f)
# 	inputs=get_inputs(orientation,position,map)
# 	print(inputs)
# 	end_0=scale*(position+orientation*inputs[0])+border
# 	end_1=scale*(position+rotation(np.pi/4,orientation)*inputs[2])+border
# 	end_2=scale*(position+rotation(-np.pi/4,orientation)*inputs[4])+border
# 	print(end_0)
# 	draw.line([(sc[0],sc[1]),tuple(end_0)],fill=laser_rgb,width=4)
# 	draw.ellipse([tuple(end_0-dcirc),tuple(end_0+dcirc)], fill=(230,100,0), outline=None)
# 	draw.line([(sc[0],sc[1]),tuple(end_1)],fill=laser_rgb,width=4)
# 	draw.ellipse([tuple(end_1-dcirc),tuple(end_1+dcirc)], fill=(230,100,0), outline=None)
# 	draw.line([(sc[0],sc[1]),tuple(end_2)],fill=laser_rgb,width=4)
# 	draw.ellipse([tuple(end_2-dcirc),tuple(end_2+dcirc)], fill=(230,100,0), outline=None)
# 	pil_f=reflect_y_axis(pil_f)
# 	draw = ImageDraw.Draw(pil_f)
# 	font = ImageFont.truetype("arial.ttf", int(40*imsize/1000))
# 	dy=0
# 	text=['left distance: '+str(inputs[2])[:4],'straight distance: '+str(inputs[0])[:4],'right distance: '+str(inputs[4])[:4]]
# 	for k in range(len(text)):
# 		font = ImageFont.truetype("arial.ttf", int(40*imsize/1000))
# 		draw.text((int(0.5*imsize),int(0.2*imsize+dy)), text[k], font=font, fill=(0,0,0))
# 		dy+=1.1*font.getsize(text[k])[1]
# 	pil_f.save('video_figures/images/car_measure_'+str(i)+'.png',format='png')

race_map=race_map_list[0]
old_score=0
for g in range(N_gen):
	print('Race: '+str(g+1))
	game=Game(race_map,cars,dt=opt.dt,n_iter=opt.n_iter,save_path=save_path)
	game.max_rounds_race_only_front_sight()
	# game.max_rounds_race_shape()
	# print(game.scores)
	best_score=np.max(game.scores)
	print(best_score)
	if g%2==0 and old_score<best_score:
		game.plot_game(imsize=int(180*np.sqrt(race_map.size)),path='gifs/shape_generation='+str(g+10+1)+'_map='+str(opt.map_size)+'_nh='+str(opt.n_h)+'.avi',car_shape=car_shape)
		old_score=best_score

	cars=game.selection_and_mutation(N_sel,N_mut,shape_mutation=False,mut_fac=1)







	
