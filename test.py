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
parser.add_argument('--n_h', type=int, default=6)
parser.add_argument('--dt', type=float, default=0.08)
parser.add_argument('--load_car', type=str2bool, default=False)
parser.add_argument('--load_dream', type=str2bool, default=False)

np.random.seed(0)
opt = parser.parse_args()
print(opt)


# img=create_image(200, 200)
# img.save('cars/car_4.png',format='png')
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
n_race_maps=1
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



# for j in range(len(race_map_list)):
# 	race_map_list[j].draw_map()
# 	print(len(race_map_list[j].skeleton))


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
else:
	for nc in range(0):
		cars.append(Car(size=0.15,model=1,grip=-0.,v_max=1000,n_inputs=3,n_h=15))
	for nc in range(n_cars):
		cars.append(Car(grip=-0.5,model=3,n_h=opt.n_h,mutate_physics=True,v_max=1))



environment_model=Environment_model(n_layers=2,depth=3,h_dim=10)
reg_model=Regression_model(n_layers=1)
optimizer= torch.optim.Adam(environment_model.parameters(), lr=0.0002)
optimizer_reg=torch.optim.Adam(reg_model.parameters(),lr=0.0002)
if opt.load_dream:
  print('reload model....')
  state_dict=torch.load('dream_models/environment_model.pkl')
  environment_model.load_state_dict(state_dict['model_state'])
  optimizer.load_state_dict(state_dict['optimizer_state'])



scores=np.zeros(n_cars)
for g in range(N_gen):
	print('Race: '+str(g+1))
	for m in range(len(race_map_list)):
		print('map: '+str(m+1))
		game=Game(race_map_list[m],cars,dt=opt.dt,n_iter=opt.n_iter,save_path=save_path)
		game.scores=scores
		data=game.max_rounds_race_shape(get_data=True)
		game.plot_game(imsize=int(40*np.sqrt(race_map_list[m].size)),path='gifs/shape_generation='+str(g+1)+'_map='+str(m+1)+'.gif',car_shape=True)
		scores=game.scores
	cars=game.selection_and_mutation(N_sel,N_mut,shape_mutation=True,mut_fac=1)
	scores=np.zeros(n_cars)
	plot_car_perspective(data[0],game,game.car_list[0])
	data=transform_data_to_alpha_and_norm(data)
	train_environment_model(environment_model,optimizer,data,n_epochs=5000,save_path='dream_models/environment_model.pkl',print_every=200,stop_loss=-90)
	ds_data,input_data,score_data=game.dream(environment_model)
	game.plot_halucination(ds_data[0][0:90],input_data[0][0:90],score_data[0][0:90],imsize=130,path='gifs/halu.gif',car=game.car_list[0],car_shape=True)

data=transform_data_to_alpha_and_norm(data)
environment_model=train_environment_model(environment_model,optimizer,data,n_epochs=1000,save_path='dream_models/environment_model.pkl',print_every=100,stop_loss=-130)

# game=Game(race_map_list[0],cars,dt=opt.dt,n_iter=opt.n_iter,save_path=save_path)
# ds_data,input_data,score_data=game.dream(environment_model)
# game.plot_halucination(ds_data[0][0:150],input_data[0][0:150],score_data[0][0:150],imsize=130,path='gifs/halu.gif',car=game.car_list[0],car_shape=True)


# for g in range(N_gen):
# 	print('Dream: '+str(g+1))
# 	game=Game(race_map_list[0],cars,dt=opt.dt,n_iter=opt.n_iter,save_path=save_path)
# 	game.dream(environment_model)
# 	game.plot_game(imsize=int(100*np.sqrt(race_map_list[0].size)),path='gifs/shape_generation='+str(g+1)+'_map='+str(0+1)+'.gif',car_shape=True)
# 	cars=game.selection_and_mutation(N_sel,N_mut,shape_mutation=True)





	
