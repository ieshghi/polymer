#parts of this have been taken from "Introduction to Normalizing Flows for Lattice Field Theory" by Albergo et. al., which can be found for free on arXiv

import base64
import io
import pickle
import numpy as np
import torch
import torch.nn as nn
print(f'TORCH VERSION: {torch.__version__}')
import packaging.version
if packaging.version.parse(torch.__version__) < packaging.version.parse('1.5.0'):
	raise RuntimeError('Torch versions lower than 1.5.0 not supported')

#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import normflows as nf
from tqdm import tqdm

sns.set_style('whitegrid')

if torch.cuda.is_available():
	torch_device = 'cuda'
	float_dtype = np.float32 # single
	torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
	torch_device = 'cpu'
	float_dtype = np.float64 # double
	torch.set_default_tensor_type(torch.DoubleTensor)
	print(f"TORCH DEVICE: {torch_device}")

# we want to sample SAWs on a 3d lattice first. To do this, treat it like sampling from a Boltzmann distribution and make a strong repulsive interaction energy?
def cont2int(steps):
	return torch.round((steps + 0.5)*6)[0] #assume steps is distributed between -1 and 1

class conform: #stores a polymer conformation
	def __init__(self,steps,initpos = torch.Tensor([0,0,0])):
		self.steps = cont2int(steps)
		self.initpos = initpos
		self.step2vec = {0:torch.Tensor([0,1,0]),1:torch.Tensor([0,-1,0]),2:torch.Tensor([1,0,0]),3:torch.Tensor([-1,0,0]),4:torch.Tensor([0,0,1]),5:torch.Tensor([0,0,-1])}    # Convention: "steps" is an (N-1)x1 array where N is the number of monomers in the chain. Each step is a number from 0-5. 0: up (y+), 1: down(y-), 2: right(x+), 3:left(x-), 4:out of the page(z+), 5:into the page(z-)
	def get_points(self): #"steps" is the array of steps in the lattice SAW. We must convert from "steps" to an array of positions, and from there evaluate the probability distribution (here we'll treat it as Boltzmann for now).
		return torch.cumsum(torch.stack([self.step2vec.get(int(i)) for i in self.steps]),dim=0) + self.initpos 

class energyfunc(nn.Module): #give a penalty = sigma for each overlapping pair of monomers
	def __init__(self,sigma = 10,bdry=False,constraints = False):
		super(energyfunc, self).__init__()
		self.bdry = bdry #if this is set to a finite number, it is the radius of the confining sphere in lattice units
		self.sigma = sigma #interaction energy for overlapping monomers, in units of K_b T. Probability of overlap should scale like exp(-sigma)
		self.constraints = constraints #for when we want to eventually include constraints from pore-c or something

	def forward(self,steps):
		energ = 0
		conf = conform(steps).get_points()
		if self.constraints:
			energ += check_constraints(conf)
		if self.bdry:
			energ += check_bdry(conf)
		energ += (conf.shape[0]-conf.unique(dim=0).shape[0])*self.sigma
		return energ

	def check_constraints(self,config):
		return 0
	def check_bdry(self,config):
		return 0

N = 1000
prior = nf.distributions.UniformGaussian(N,range(0,N),torch.ones(N))

num_layers = 20
flow_layers = []
for i in range(num_layers):
    # Neural network with two hidden layers having 64 units each
    # Last layer is initialized by zeros making training more stable
    param_map = nf.nets.MLP([1, 64, 64, N], init_zeros=True)
    # Add flow layer
    flow_layers.append(nf.flows.AffineCouplingBlock(param_map))

model = nf.NormalizingFlow(prior,flow_layers)

max_iter = 20000
anneal_iter = 10000
show_iter = 1000
loss_hist = np.array([])

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
config = prior.sample()
for it in tqdm(range(max_iter)):
    optimizer.zero_grad()
    loss = energyfunc(config)
    if ~(torch.isnan(loss) | torch.isinf(loss)):
        loss.backward()
        optimizer.step()
    
    loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
    
# Plot loss
plt.figure(figsize=(10, 10))
plt.plot(loss_hist, label='loss')
plt.legend()
plt.savefig('~/public_html/polygen/bla.png') 
