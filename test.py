from utils import *
import random



random.seed(0)

race_map=Map(size=3)
car=Car(size=0.1)
game=Game(race_map,car,dt=0.02,n_iter=100)
game.simple_race()
game.plot_game(imsize=1000)
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