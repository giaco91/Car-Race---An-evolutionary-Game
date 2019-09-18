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
parser.add_argument('--load_path', default='best_cars/', help='path to load a trained car')
parser.add_argument('--save_path', default='best_cars/', help='path to save the best car')
parser.add_argument('--map_size', type=int, default=3, help='size of the map')
parser.add_argument('--model', type=int, default=3, help='model of the car, number depending on how many models have been drawn')
parser.add_argument('--n_sel', type=int, default=3, help='number of selected parents for the next generation')
parser.add_argument('--n_mut', type=int, default=50, help='number of mutated children in every generation')
parser.add_argument('--n_gen', type=int, default=10, help='over how many generations do you want to train?')
parser.add_argument('--n_iter', type=int, default=150, help='the maximum number of iteration steps (discrete time) for each game. If cars become good, it makes this number should become larger')
parser.add_argument('--n_h', type=int, default=3, help='number of neurons in the hidden layer')
parser.add_argument('--dt', type=float, default=0.08, help='time resolution. If small, the n_iter argument should be made larger (for an equailly long game)')
parser.add_argument('--load_car', type=str2bool, default=False, help='do you want to load a trained car?')
parser.add_argument('--shape_mutation', type=str2bool, default=True, help='do you want to mutate (in addition to the neural network) also the cars shape?')
parser.add_argument('--avi', type=str2bool, default=False, help='do you want have an .avi video output instead of a gif?')
parser.add_argument('--video_quality', type=float, default=1, help='the larger the number, the more pixels are calculated for the frames (and the longer it takes to render)')
opt = parser.parse_args()
print(opt)

#set seed for reproducability
np.random.seed(0)

if opt.avi:
	video_ending='.avi'
else:
	video_ending='.gif'

if opt.save_path=='best_cars/':
	save_path=opt.save_path+'nh='+str(opt.n_h)+'.pkl'
else:
	save_path=opt.save_path
if opt.load_path=='best_cars/':
	load_path=opt.load_path+'nh='+str(opt.n_h)+'.pkl'
else:
	load_path=opt.load_path
	print('load car from: '+str(opt.load_path))

#---load car---
cars=[]
if opt.load_car:
	print('loading car ...')
	with open(load_path, 'rb') as f:
	    best_car = pickle.load(f)
	    cars.append(deepcopy(best_car))
	    for nc in range(opt.n_mut-1):
	    	cars.append(deepcopy(best_car))
	    	cars[-1].mutation(shape_mutation=opt.shape_mutation)
else:
	for nc in range(opt.n_mut):
		cars.append(Car(grip=-1,model=opt.model,n_h=opt.n_h,n_inputs=3,mutate_physics=opt.shape_mutation,v_max=1,size=0.15,F_max=2000))

#---make directionries ---
if not os.path.exists(opt.save_path):
    os.mkdir(opt.save_path)
if not os.path.exists('videos/'):
    os.mkdir('videos/')

#create a random race map----
race_map=Map(size=opt.map_size)
old_score=0
for g in range(opt.n_gen):
	print('Race: '+str(g+1))
	#play a racing game ----
	game=Game(race_map,cars,dt=opt.dt,n_iter=opt.n_iter,save_path=save_path)
	game.max_rounds_race_shape()
	best_score=np.max(game.scores)
	if old_score<best_score:
		#make video of racing game---
		# game.plot_game(imsize=int(100*np.sqrt(race_map.size)),path='videos/generation='+str(g+10+1)+'_map='+str(opt.map_size)+'bestscore='+str(best_score)[:5]+'_nh='+str(opt.n_h)+'.avi',car_shape=opt.shape_mutation)
		game.plot_game(imsize=int(opt.video_quality*100*np.sqrt(race_map.size)),path='videos/generation='+str(g+1)+'_map='+str(opt.map_size)+'bestscore='+str(best_score)[:5]+'_nh='+str(opt.n_h)+video_ending,car_shape=opt.shape_mutation)
		old_score=best_score
	#selection and mutation step---
	cars=game.selection_and_mutation(opt.n_sel,opt.n_mut,shape_mutation=opt.shape_mutation,mut_fac=1)







	
