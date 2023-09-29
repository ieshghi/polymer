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

sns.set_style('whitegrid')

if torch.cuda.is_available():
    torch_device = 'cuda'
    float_dtype = np.float32 # single
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch_device = 'cpu'
    float_dtype = np.float64 # double
    torch.set_default_tensor_type(torch.DoubleTensor)
    torch.set_num_threads(10)

print(f"TORCH DEVICE: {torch_device}")

def smoothstep(x): #logistic function with a given steepness
    k = 20
    return 1/(1+torch.exp(-k*(x-1)))

class rescale_layer(nn.Module): #want model outputs to be exactly in the range [-0.5,0.5] so we use this sigmoid function, but rescaled so that for most inputs near 0 it doesn't warp them too much
    def __init__(self):
        super().__init__()

    def forward(self,z):
        return z,torch.sigmoid(z*4) - 0.5

    def inverse(self,z):
        return z_,torch.logit(z+0.5)/4

class energyfunc_loss(nn.Module): #give a penalty = sigma for each overlapping pair of monomers
    def __init__(self,N,sigma = 10,lcol = 0.2,ltouch = 0.5,constraints = False): #assume step size is unit length, lcol is collision distance
        super(energyfunc_loss, self).__init__()
        self.sigma = sigma #interaction energy for overlapping monomers, in units of K_b T. Probability of overlap should scale like exp(-sigma)
        self.constraints = constraints #for when we want to eventually include constraints from pore-c or something
        self.ltouch = ltouch #distance at which contacts are registered
        self.lcol = lcol #distance below which monomers are not allowed to go
        self.N = N
        self.contact_logexp = get_contact_exp().log()

    def forward(self,confdata):
        polymers = randtopoly(confdata)
        distances = torch.cdist(polymers,polymers)
        
        dists = torch.triu(distances)
        dists = dists[dists>0]
#        collision_energy = self.sigma*torch.sum(1-smoothstep(dists/self.lcol)) #punish collisions with a smoothed step function
        fractal_loss = contacts_kld(torch.sum(1-smoothstep(distances/self.ltouch),axis=0).triu()) #count contacts with a smoothed step function

#        tot_energy = collision_energy + fractal_loss
        tot_energy = fractal_loss
        return tot_energy

    def check_constraints(self,positions):
        return 0

    def get_contact_exp(self):
        inds = torch.arange(self.N)
        smat = torch.zeros(N,N)
		for i in range(N):
			smat[i,:] = torch.abs(int-i)

		pmat = (1/smat).triu()
        return pmat/torch.sum(pmat)        

    def contacts_kld(self,contacts):
        probs = contacts/torch.sum(contacts) 
        return torch.nansum(probs*(probs.log()-self.contact_logexp))

def randtopoly(confdata):
    r1 = confdata[:,::2].transpose(0,1) #even indices give (phi/pi - 1)/2
    r2 = confdata[:,1::2].transpose(0,1) #odd indices give -cos(theta)/2

    n = r1.shape[1]
    ph = (2*r1 + 1)*torch.pi
    th = torch.acos(-2*r2) 
    x = torch.cat([torch.zeros(1,n),torch.sin(th)*torch.cos(ph)])
    y = torch.cat([torch.zeros(1,n),torch.sin(th)*torch.sin(ph)])
    z = torch.cat([torch.ones(1,n),torch.cos(th)])

    steps = torch.stack([x,y,z])
    pos_arr = torch.cumsum(steps,axis=0).transpose(0,1)
    pos_arr = torch.cat([torch.zeros(1,3,n),pos_arr])
    return pos_arr.transpose(0,2).transpose(1,2) #return a tensor of shape (BxNxD) where B is batch size, N is length of the polymer, and D is space dimension (=3).

def get_raw_model_and_loss(N,num_layers=5,inner_layer_dim=64): 
    #given a polymer size N, returns the network to be trained as well as the loss function for that polymer shape

    dof = 2*(N-2) # The first monomer is at the origin, and the second is at (0,0,1) , then 2 dof for the 3d orientation of (N-2) unit vectors in a polymer with N monomers
    prior = nf.distributions.UniformGaussian(dof,range(dof),torch.ones(dof)) #get random numbers uniformly distributed between -1/2 and 1/2
    flow_layers = []
    for i in range(num_layers):
        param_map = nf.nets.MLP([dof//2, inner_layer_dim, inner_layer_dim, dof], init_zeros=True) #should probably update this to obey spherical symmetry
        flow_layers.append(nf.flows.AffineCouplingBlock(param_map,split_mode="checkerboard"))
        flow_layers.append(rescale_layer())
    model = nf.NormalizingFlow(prior,flow_layers)

    loss_f = energyfunc_loss(N)
    return model,loss_f

def run(N = 1000,ifsave=True,nam='model',use_pretrained=False,load_nam='model'):
    
    if use_pretrained:
        model = torch.load('models/'+nam+'.pickle')
        loss_func = energyfunc_loss()
    else:
        model,loss_func = get_raw_model_and_loss(N)
    
    model = model.to(torch_device)

    max_iter = 10000
    show_iter = 100
    num_samples = 100
    loss_hist = np.array([])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
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
