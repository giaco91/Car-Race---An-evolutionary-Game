from utils import *



cars=[]
N_sel=3
N_mut=30
N_gen=19
n_cars=N_mut
plot_every=3

race_map=Map(size=5)
race_map.draw_map()
for nc in range(int(n_cars/2)):
	cars.append(Car(size=0.15,model=1,grip=2))
for nc in range(n_cars-int(n_cars/2)):
	cars.append(Car(size=0.18,model=2,grip=2))


for g in range(N_gen):
	print('Race '+str(g+1))
	game=Game(race_map,cars,dt=0.05,n_iter=300)
	game.max_rounds_race()
	cars=game.selection_and_mutation(N_sel,N_mut)
	if np.mod(g,plot_every)==0:
		game.plot_game(imsize=300,path='gifs/car_race_'+str(g+1)+'.gif')
game.plot_game(imsize=400,path='gifs/car_race_final_version.gif')

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