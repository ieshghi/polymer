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
pool = mp.Pool(processes = 5)

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
    torch.set_num_threads(10)

def conf_compare(conf1,conf2,nam = 'plot'):
    plt.close('all')
    loss_func = energyfunc_loss()
    n1 = conf1.shape[0]
    conf1_list = [conf1[i,:] for i in range(n1)]
    n2 = conf2.shape[0]
    conf2_list = [conf2[i,:] for i in range(n2)]
    poly1 = list(map(loss_func.get_poly,conf1_list))
    poly2 = list(map(loss_func.get_poly,conf2_list))
    msd1_list = list(map(msd,poly1))
    msd2_list = list(map(msd,poly2))
    n1_step = msd1_list[0].shape[0]
    n2_step = msd2_list[0].shape[0]
    t1 = np.arange(n1_step)
    t2 = np.arange(n2_step)
    msd1_mean = np.nanmean(msd1_list,axis=0)
    msd2_mean = np.nanmean(msd2_list,axis=0)
#    for i in range(n_samp):
#        plt.loglog(t,msd_list[i])
    plt.loglog(t1,msd1_mean,'-',label='Data 1')
    plt.loglog(t2,msd2_mean,'--',label='Data 2')
    plt.legend()
    plt.savefig('/gpfs/commons/home/ieshghi/public_html/polygen/'+nam+'.png')
    return msd1_mean,msd2_mean

def conf_analyze(confdata,nam = 'plot'):
    plt.close('all')
    loss_func = energyfunc_loss()
    n_samp = confdata.shape[0]
    conf_list = [confdata[i,:] for i in range(n_samp)]
    poly_list = list(map(loss_func.get_poly,conf_list))
    msd_list = list(map(msd,poly_list))
    n_step = msd_list[0].shape[0]
    t = np.arange(n_step)
    msd_mean = np.mean(msd_list,axis=0)
#    for i in range(n_samp):
#        plt.loglog(t,msd_list[i])
    plt.loglog(t,msd_mean,'-',label='Data')
    plt.loglog(t,10*t,'--',label='Random Walk')
    plt.loglog(t,10*t**(5/3),'--',label='SAW')
    plt.legend()
    plt.savefig('/gpfs/commons/home/ieshghi/public_html/polygen/'+nam+'.png')
    return msd_list
    
def msd(xyz):
    n = xyz.shape[0]
    shifts = np.arange(0,n-1)
    msds = np.zeros(shifts.size)
    msds_std = np.zeros(shifts.size)
    xyz = xyz.detach().numpy()

    for i, shift in enumerate(shifts):
        diffs = xyz[:-shift if shift else None] - xyz[shift:]
        sqdist = np.square(diffs).sum(axis=1)
        msds[i] = sqdist.mean()
   #     msds_std[i] = sqdist.std(ddof=1)

    return msds

class rescale_layer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,z):
        return z,torch.sigmoid(z*4) - 0.5

    def inverse(self,z):
        return z_,torch.logit(z+0.5)/4

class energyfunc_loss(nn.Module): #give a penalty = sigma for each overlapping pair of monomers
    def __init__(self,sigma = 10,lcol = 0.5,bdry=False,constraints = False): #assume step size is unit length, lcol is collision distance
        super(energyfunc_loss, self).__init__()
        self.bdry = bdry #if this is set to a finite number, it is the radius of the confining sphere in lattice units
        self.sigma = sigma #interaction energy for overlapping monomers, in units of K_b T. Probability of overlap should scale like exp(-sigma)
        self.constraints = constraints #for when we want to eventually include constraints from pore-c or something
        self.lcol = lcol

    def forward(self,confdata):
#        return torch.sum(confdata**2)
        n = confdata.shape[0]
        energy = self.evaluate_chain(confdata[0,:])
        if n>1:
            for i in range(1,n):
                energy += self.evaluate_chain(confdata[i,:])
        return energy

    def evaluate_chain(self,single_conf):
        polypos = self.get_poly(single_conf)
        energ = self.sigma*self.collisions(polypos)
        if self.constraints:
            energ += self.check_constraints(conf)
        if self.bdry:
            energ += self.check_bdry(conf)
        return energ

    def check_constraints(self,config):
        return 0
    def check_bdry(self,config):
        return 0
    def collisions(self,positions):
        dists = torch.cdist(positions,positions)
#        dists = dists[dists>0]
#        return torch.sum((self.lcol/dists)**12-(self.lcol/dists)**6) #punish overlap within distance lcol with a lennard-jones potential (this is a bit aggressive maybe)
        return torch.sum(torch.exp(-(dists/self.lcol)**2)) #punish overlap within distance lcol with a Gaussian
        #return torch.sum(dists**2)

    def get_poly(self,confdata):
        init_pos = 0*confdata[1:4].reshape(1,3)
        confdata = confdata[4:] + 0.5 #shift the uniform random numbers up so they range [0,1]
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

def run(ifsave=True,nam='model',use_pretrained=False,load_nam='model'):
    
    if use_pretrained:
        model = torch.load('models/'+nam+'.pickle')
    else:
        N = 1000
        dof = 4 + 2*(N-1) # 3 degrees of freedom for the position of the first monomer, then 2 dof for the 3d orientation of (N-1) unit vectors in an N-step SAW + 1 fake dof just so it's an even number
        prior = nf.distributions.UniformGaussian(dof,range(dof),torch.ones(dof)) #get random numbers uniformly distributed between -1/2 and 1/2
        #
        num_layers = 5
        flow_layers = []
        for i in range(num_layers):
            # Last layer is initialized by zeros making training more stable
            param_map = nf.nets.MLP([dof//2, 64, 64, dof], init_zeros=True)
            # Add flow layer
            flow_layers.append(nf.flows.AffineCouplingBlock(param_map,split_mode="checkerboard"))
            flow_layers.append(rescale_layer())
            #flow_layers.append(nf.flows.Permute(2, mode='swap'))
        
        model = nf.NormalizingFlow(prior,flow_layers)
    
    max_iter = 100000
    show_iter = 100
    num_samples = 16
    loss_hist = np.array([])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_func = energyfunc_loss()
    i = 0
    for it in tqdm(range(max_iter)):
        i += 1
        optimizer.zero_grad()
        config = model.q0.sample(num_samples)
        output = model(config)
        loss = loss_func.forward(output)
        if i%show_iter == 0:
            print(loss)
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()
        
        loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())

    if ifsave:
        torch.save(model,'models/'+nam+'.pickle')
        
        # Plot loss
        plt.figure(figsize=(10, 10))
        plt.plot(loss_hist, label='loss')
        plt.legend()
        plt.savefig('/gpfs/commons/home/ieshghi/public_html/polygen/loss_hist.png')
    return loss_hist,model

run(nam='freesaw_1000beads',use_pretrained=True,load_nam='freesaw_1000beads')
