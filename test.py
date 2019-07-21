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
parser.add_argument('--n_h', type=int, default=5)
parser.add_argument('--dt', type=float, default=0.08)
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

def get_loss(m_hat,r_hat,m,r,batchSize,seq_lengths):
	loss=0
	for bs in range(batchSize):
		dm_bs=m_hat[bs,:seq_lengths[bs],:]-m[bs,:seq_lengths[bs],:]
		dr_bs=r_hat[bs,:seq_lengths[bs],:]-r[bs,:seq_lengths[bs],:]
		loss+=(torch.sum(torch.mul(dm_bs,dm_bs))+torch.sum(torch.mul(dr_bs,dr_bs)))/int(seq_lengths[bs])
	return loss

def get_reg_loss(m_hat,r_hat,m,r):
	L=m_hat.size(0)
	d_m=m_hat-m
	d_r=r_hat-r
	loss=(torch.sum(torch.mul(d_m,d_m))+torch.sum(torch.mul(d_r,d_r)))/L
	return loss

environment_model=Environment_model(n_layers=1)
reg_model=Regression_model(n_layers=1)
optimizer= torch.optim.Adam(environment_model.parameters(), lr=0.001)
optimizer_reg=torch.optim.Adam(reg_model.parameters(),lr=0.001)

scores=np.zeros(n_cars)
for g in range(N_gen):
	print('Race: '+str(g+1))
	for m in range(len(race_map_list)):
		print('map: '+str(m+1))
		game=Game(race_map_list[m],cars,dt=opt.dt,n_iter=opt.n_iter,save_path=save_path)
		game.scores=scores
		data=game.max_rounds_race_shape(get_data=True)
		scores=game.scores
		l=len(data[0])
		print(l)
		car_data=np.zeros((l,6))
		for i in range(l):
			car_data[i,:]=data[0][i]
		game.plot_halucination(car_data[:,4:],car_data[:,0:3],car_data[:,3],imsize=200,path='gifs/halu.gif',car=game.car_list[0],car_shape=True)

		game.plot_game(imsize=int(100*np.sqrt(race_map_list[m].size)),path='gifs/shape_generation='+str(g+1)+'_map='+str(m+1)+'.gif',car_shape=True)
	cars=game.selection_and_mutation(N_sel,N_mut,shape_mutation=True)
	scores=np.zeros(n_cars)

def get_reg_data(data):
	l_tot=0
	for i in range(len(data)):
		l_tot+=len(data[i])
	x=torch.zeros(l_tot,5)
	t=torch.zeros(l_tot,4)
	current_l=0
	for i in range(len(data)):
		x[current_l:current_l+len(data[i]),:]=torch.from_numpy(np.asarray(data[i])[:,np.array([0,1,2,4,5])])
		t[current_l:current_l+len(data[i]),:4]=torch.from_numpy(np.asarray(data[i]))[:,:4]
		current_l+=len(data[i])
	return x,t

# n_epochs=10000
# x,t=get_reg_data(data)
# print_every=500
# for i in range(n_epochs):
# 	optimizer_reg.zero_grad()
# 	m,r=reg_model(x)
# 	loss_reg=get_reg_loss(m,r,t[:,0:3],t[:,3].unsqueeze(1))
# 	loss_reg.backward()
# 	optimizer_reg.step()
# 	if i%print_every==0:
# 		print('reg loss: '+str(loss_reg.item()))

# 	optimizer.zero_grad()
# 	packed_sequences=pack_sequences(data)
# 	out_m, out_r, h_rare=environment_model(packed_sequences,n_cars)
# 	unpacked_sequences,seq_lengths=unpack_sequences(packed_sequences)
# 	m=unpacked_sequences[:,:,0:3]
# 	r=unpacked_sequences[:,:,3].unsqueeze(2)
# 	loss=get_loss(out_m,out_r,m,r,n_cars,seq_lengths)
# 	loss.backward()
# 	optimizer.step()
# 	if i%print_every==0:
# 		print('rnn loss: '+str(loss.item()))

	
