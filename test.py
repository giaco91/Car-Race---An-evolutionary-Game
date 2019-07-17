from utils import *
from car import *
from game import *
from map import *


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
parser.add_argument('--n_h', type=int, default=5)
parser.add_argument('--load_car', type=str2bool, default=False)

np.random.seed(0)
opt = parser.parse_args()
print(opt)

load_car=opt.load_car
cars=[]
N_sel=opt.n_sel
N_mut=opt.n_mut
N_gen=opt.n_gen
n_cars=N_mut
plot_every=opt.plot_every
# save_path1='best_cars/front_sight_nh=15.pkl'
save_path=opt.save_path+'nh='+str(opt.n_h)+'.pkl'
# save_path2='best_cars/low_grip.pkl'
load_path=opt.load_path+'nh='+str(opt.n_h)+'.pkl'
# race_map=Map(size=4)
# race_map.draw_map()

# race_map_list=[Map(size=3),Map(size=4),Map(size=5),Map(size=6),Map(size=7),Map(size=8)]
# race_map_list=[Map(size=4),Map(size=4),Map(size=4),Map(size=4),Map(size=4)]
race_map_list=[]
n_race_maps=1
size=opt.map_size
if size==5:
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


# for j in range(len(race_map_list)):
# 	race_map_list[j].draw_map()
	# print(len(race_map_list[j].skeleton))


# raise ValueError('')
#load
if load_car:
	print('loading car ...')
	with open(load_path, 'rb') as f:
	    best_car = pickle.load(f)
	    cars.append(deepcopy(best_car))
	    for nc in range(n_cars-1):
	    	cars.append(deepcopy(best_car))
	    	cars[-1].mutation(shape_mutation=True)
	    	# cars[-1].grip=-1
	    	# cars[-1].v_max=100
else:
	for nc in range(0):
		cars.append(Car(size=0.15,model=1,grip=-0.,v_max=1000,n_inputs=3,n_h=15))
	for nc in range(n_cars):
		# cars.append(Car(size=0.18,model=2,grip=-1.2,v_max=1000,n_inputs=3,n_h=15))
		cars.append(Car(grip=-0.5,n_h=opt.n_h,mutate_physics=True,v_max=1))
		# print(cars[-1].v_max)
		# print(cars[-1].grip)

# proto_car=[]
# new_car=Car(size=0.15,model=1,grip=1,v_max=5,n_inputs=3,n_h=3)
# with open(save_path2, 'rb') as f:
#     best_car = pickle.load(f)
# new_car.a_weights=best_car.a_weights
# new_car.c_weights=best_car.c_weights
# new_car.h_weights[0:2,:]=best_car.h_weights[0:2,:]
# new_car.h_weights[2:,:]=best_car.h_weights[3:,:][0::2,:]
# for nc in range(n_cars):
# 	cars.append(deepcopy(new_car))
# 	cars[-1].mutation()
# 	cars[-1].grip=-1
# 	cars[-1].v_max=100



scores=np.zeros(n_cars)
for g in range(N_gen):
	print('Race: '+str(g+1))
	for m in range(len(race_map_list)):
		print('map: '+str(m+1))
		game=Game(race_map_list[m],cars,dt=0.08,n_iter=opt.n_iter,save_path=save_path)
		game.scores=scores
		# game.max_rounds_race_only_front_sight()
		data=game.max_rounds_race_shape(get_data=True)
		scores=game.scores
		game.plot_game(imsize=int(100*np.sqrt(race_map_list[m].size)),path='gifs/shape_generation='+str(g+1)+'_map='+str(m+1)+'.gif',car_shape=True)
		# game.plot_game(imsize=int(90*np.sqrt(race_map_list[m].size)),path='gifs/front_generation='+str(g+1+24)+'_map='+str(m+1)+'.gif')
	cars=game.selection_and_mutation(N_sel,N_mut,shape_mutation=True)
	scores=np.zeros(n_cars)


	

# for g in range(N_gen):
# 	print('Race '+str(g+1))
# 	# game=Game(Map(size=5),cars,dt=0.1,n_iter=300,save_path=save_path1)
# 	game=Game(race_map,cars,dt=0.1,n_iter=300,save_path=save_path1)
# 	game.max_rounds_race_only_front_sight()
# 	cars=game.selection_and_mutation(N_sel,N_mut)
# 	if np.mod(g,plot_every)==0:
# 		game.plot_game(imsize=200,path='gifs/generation_front'+str(g+1)+'.gif')

# game.plot_game(imsize=400,path='gifs/car_race_final_version.gif')

# p1=np.array([0.3,0.5])
# d1=np.array([1,0])
# p2=np.array([-1,-2])
# d2=np.array([2,2])
# p3=np.array([1,2])
# p4=np.array([1,-2])
# p5=np.array([2,-2])

# segment_list=[p3,p4,p5]
# print(race_map.skeleton)

# t_pos,t_neg=race_map.closest_intersection(p1,d1,race_map.border2)
# print(t_pos)
# print(t_neg)
# print(p1+d1*t_pos)
# print(p1+d1*t_neg)

# t=intersection_of_two_lines(p1,d1,segment_list[0],segment_list[1]-segment_list[0])
# print(t)
# s=p1+t[0]*d1
# print(s)