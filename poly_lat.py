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
import itertools
import multiprocessing as mp
from sklearn.metrics import pairwise_distances
num_c = mp.cpu_count()
pool = mp.Pool(processes = num_c)

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

class energyfunc(nn.Module): #give a penalty = sigma for each overlapping pair of monomers
    def __init__(self,sigma = 10,lcol = 0.1,bdry=False,constraints = False): #assume step size is unit length, lcol is collision distance
        super(energyfunc, self).__init__()
        self.bdry = bdry #if this is set to a finite number, it is the radius of the confining sphere in lattice units
        self.sigma = sigma #interaction energy for overlapping monomers, in units of K_b T. Probability of overlap should scale like exp(-sigma)
        self.constraints = constraints #for when we want to eventually include constraints from pore-c or something
        self.lcol = lcol

    def forward(self,confdata):
        energ = 0
        polypos = self.get_poly(confdata)
        if self.constraints:
            energ += self.check_constraints(conf)
        if self.bdry:
            energ += self.check_bdry(conf)
        energ += self.sigma*self.collisions(polypos)
        return energ

    def check_constraints(self,config):
        return 0
    def check_bdry(self,config):
        return 0
    def collisions(self,positions):
        dists = torch.Tensor(pairwise_distances(positions))
        return torch.sum(1/(1+torch.exp(-dists/self.lcol))) #punish overlap within distance lcol with a sigmoid function
    def get_poly(self,confdat):
        confdata = confdat[0]
        init_pos = confdata[0:3].reshape(1,3)
        confdata = confdata[3:] + 0.5 #shift the uniform random numbers up so they range [0,1]
        r1 = confdata[::2]
        r2 = confdata[1::2]
        steps = randtovec(r1,r2)
        steps = torch.cat([init_pos,steps.transpose(0,1)])
        pos_arr = torch.cumsum(steps,axis=0)
        return pos_arr

def randtovec(r1,r2):
    ph = (r1)*2*torch.pi
    th = torch.acos(1-2*(r2)) #nf.dists module makes uniform distributions on [-0.5,0.5]
    x = torch.sin(th)*torch.cos(ph)
    y = torch.sin(th)*torch.sin(ph)
    z = torch.cos(th)
    return torch.stack([x,y,z])

N = 100
dof = 3+2*(N-1) # 3 degrees of freedom for the position of the first monomer, then 2 dof for the 3d orientation of (N-1) unit vectors in an N-step SAW
prior = nf.distributions.UniformGaussian(dof,range(dof),torch.ones(dof)) #get random numbers uniformly distributed between -1/2 and 1/2
#
num_layers = 20
flow_layers = []
for i in range(num_layers):
    # Neural network with two hidden layers having 64 units each
    # Last layer is initialized by zeros making training more stable
    param_map = nf.nets.MLP([1, 64, 64, dof], init_zeros=True)
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
    loss = np.exp(-energyfunc(config))
    if ~(torch.isnan(loss) | torch.isinf(loss)):
        loss.backward()
        optimizer.step()
    
    loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
    
## Plot loss
#plt.figure(figsize=(10, 10))
#plt.plot(loss_hist, label='loss')
#plt.legend()
#plt.savefig('~/public_html/polygen/bla.png') 
