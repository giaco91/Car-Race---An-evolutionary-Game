from utils import *
from copy import deepcopy

np.random.seed(8)

load_car=True
cars=[]
N_sel=3
N_mut=30
N_gen=10
n_cars=N_mut
plot_every=1
save_path1='best_cars/front_sight_nh=15.pkl'
# save_path2='best_cars/low_grip.pkl'
load_path='best_cars/front_nh=15.pkl'
# race_map=Map(size=4)
# race_map.draw_map()


# race_map_list=[Map(size=3),Map(size=4),Map(size=5),Map(size=6),Map(size=7),Map(size=8)]
# race_map_list=[Map(size=4),Map(size=4),Map(size=4),Map(size=4),Map(size=4)]
race_map_list=[]
n_race_maps=1
size=5
min_points=20
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
	race_map_list[j].draw_map()
	print(len(race_map_list[j].skeleton))


# raise ValueError('')
#load
if load_car:
	with open(load_path, 'rb') as f:
	    best_car = pickle.load(f)
	    cars.append(deepcopy(best_car))
	    for nc in range(n_cars-1):
	    	cars.append(deepcopy(best_car))
	    	cars[-1].mutation()
	    	# cars[-1].grip=-1
	    	# cars[-1].v_max=100
else:
	for nc in range(0):
		cars.append(Car(size=0.15,model=1,grip=-1.2,v_max=1000,n_inputs=3,n_h=15))
	for nc in range(n_cars):
		cars.append(Car(size=0.18,model=2,grip=-1.2,v_max=1000,n_inputs=3,n_h=15))

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
best_score=0
mut_fac=1
for g in range(N_gen):
	print('Race: '+str(g+1))
	for m in range(len(race_map_list)):
		print('map: '+str(m+1))
		game=Game(race_map_list[m],cars,dt=0.13,n_iter=400,save_path=save_path1)
		game.scores=scores
		game.max_rounds_race_only_front_sight()
		scores=game.scores
		game.plot_game(imsize=int(90*np.sqrt(race_map_list[m].size)),path='gifs/front_generation='+str(g+1+22)+'_map='+str(m+1)+'.gif')
	if best_score>=np.max(scores):
		mut_fac*=2
		print('updated_mut_fac='+str(mut_fac))
	else:
		mut_fac=1
		best_score=np.max(scores)
	cars=game.selection_and_mutation(N_sel,N_mut,mut_fac=mut_fac)
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