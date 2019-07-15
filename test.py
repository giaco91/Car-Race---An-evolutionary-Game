from utils import *
from copy import deepcopy

load_car=True
cars=[]
N_sel=2
N_mut=20
N_gen=10
n_cars=N_mut
plot_every=1


race_map=Map(size=3)
# race_map_list=[Map(size=3),Map(size=4),Map(size=5),Map(size=6),Map(size=7),Map(size=8)]
race_map_list=[Map(size=5)]
# race_map.draw_map()
if load_car:
	with open('best_cars/low_grip.pkl', 'rb') as f:
	    best_car = pickle.load(f)
	    for nc in range(n_cars):
	    	cars.append(deepcopy(best_car))
	    	cars[-1].mutation()
	    	cars[-1].grip=-1.5
	    	cars[-1].v_max=10

else:
	for nc in range(1):
		cars.append(Car(size=0.15,model=1,grip=1,v_max=5))
	for nc in range(n_cars-1):
		cars.append(Car(size=0.18,model=2,grip=1,v_max=2))

scores=np.zeros(n_cars)
for g in range(N_gen):
	print('Race: '+str(g+1))
	for m in range(len(race_map_list)):
		print('map: '+str(m+1))
		game=Game(race_map_list[m],cars,dt=0.05,n_iter=400)
		game.scores=scores
		game.max_rounds_race()
		scores=game.scores
		# print(scores)
		game.plot_game(imsize=200,path='gifs/generation='+str(g+1)+'_map='+str(m+1)+'.gif')
	cars=game.selection_and_mutation(N_sel,N_mut)
	scores=np.zeros(n_cars)

	

# for g in range(N_gen):
# 	print('Race '+str(g+1))
# 	game=Game(race_map,cars,dt=0.05,n_iter=800)
# 	game.max_rounds_race()
# 	cars=game.selection_and_mutation(N_sel,N_mut)
# 	if np.mod(g,plot_every)==0:
# 		game.plot_game(imsize=300,path='gifs/generation_'+str(g+1)+'.gif')

# game.plot_game(imsize=400,path='gifs/car_race_final_version.gif')

p1=np.array([0.3,0.5])
d1=np.array([1,0])
p2=np.array([-1,-2])
d2=np.array([2,2])
p3=np.array([1,2])
p4=np.array([1,-2])
p5=np.array([2,-2])

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