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
    torch.set_num_threads(20)

print(f"TORCH DEVICE: {torch_device}")

def smoothstep(x,k=10): #logistic function with a given steepness
    return (1 - 1/(1+torch.exp(-k*(x-1))))

class energyfunc_loss(nn.Module): #give a penalty = sigma for each overlapping pair of monomers
    def __init__(self,N,k = 10,k_theta = 1,sigma = 1,fractal_sigma = 10,lcol = 0.1,ltouch = 0.5,constraints = False): #assume step size is unit length, lcol is collision distance
        super(energyfunc_loss, self).__init__()
        self.k = k #spring stiffness
        self.k_theta = k_theta #angle stiffness
        self.sigma = sigma #interaction energy for overlapping monomers, in units of K_b T. Probability of overlap should scale like exp(-sigma)
        self.fractal_sigma = fractal_sigma #
        self.constraints = constraints #for when we want to eventually include constraints from pore-c or something
        self.ltouch = ltouch #distance at which contacts are registered
        self.lcol = lcol #distance below which monomers are not allowed to go
        self.N = N
        self.triu_inds = torch.triu_indices(N,N,offset=2)
        self.contact_logexp = self.get_contact_exp().log()
        self.eps = 1e-12
        self.smat = self.get_svals()

    def forward(self,confdata):
        num_samples = confdata.shape[0]
        polymers,steps = randtopoly(confdata)
        distances = torch.cdist(polymers,polymers)
#        cos_angles = self.get_angles(steps) 
        tridists = distances[:,self.triu_inds[0],self.triu_inds[1]]
       
        spring_energy = self.k*torch.mean((torch.norm(steps,dim=2)-1)**2)

        angle_energy = 0#self.k_theta*torch.mean(1-cos_angles)

#        batchmean_collisions = torch.mean(smoothstep(tridists/self.lcol),axis=0) #punish collisions with a smoothed step function

        collision_energy = 0#self.sigma*torch.sum(batchmean_collisions)

        contacts = (self.eps + torch.sum(smoothstep(tridists/self.ltouch),axis=0))
        #fractal_loss = self.fractal_sigma*self.contacts_kld(contacts)
        fractal_loss = self.fractal_sigma*self.fractal_polov_energy(tridists)

        tot_loss = spring_energy + angle_energy + collision_energy + fractal_loss
        return tot_loss

    def check_constraints(self,positions):
        return 0
    
    def fractal_polov_energy(self,tridists): #https://arxiv.org/abs/1707.07153 shows that with quadratic interactions any fBm fractal dimension can be achieved
        gamma = 8/3 
        triu_s = self.smat[self.triu_inds[0],self.triu_inds[1]]
        return torch.mean((triu_s*tridists**2).flatten())

    def get_contact_exp(self):
        smat = self.get_svals()
        pmat = (1.0/smat).triu(diagonal=2)
        pmat_norm = (pmat/torch.sum(pmat))
        return pmat_norm[self.triu_inds[0],self.triu_inds[1]]

    def get_svals(self):
        inds = torch.arange(self.N)
        smat = torch.zeros(self.N,self.N)
        for i in range(self.N):
            smat[i,:] = torch.abs(inds-i)
        return smat

    def get_angles(self,steps):
        incoming = steps[:,:-1,:]
        outgoing = steps[:,1:,:]
        in_hat = nn.functional.normalize(incoming,dim=2)
        out_hat = nn.functional.normalize(outgoing,dim=2)
          
        cos_th = torch.sum(in_hat*out_hat,axis=2)

        return cos_th.flatten()

    def contacts_kld(self,contacts):
        probs = contacts/torch.sum(contacts) 
        kld_persite = (probs*(probs.log()-self.contact_logexp))
        return torch.sum(kld_persite)

def randtopoly(confdata):#data comes in with size BxNxD
    dim = 3
    N = confdata.shape[1]//(dim)+2
    nb = confdata.shape[0]

    steps = confdata.reshape(nb,N-2,3).transpose(0,1) #switch over to NxBxD
    steps = torch.cat([torch.zeros(1,nb,dim),steps])
    
    steps[0,:,2] = 1
    
    poly = torch.cat([torch.zeros(1,nb,dim),steps.cumsum(axis=0)])
    return poly.transpose(0,1),steps.transpose(0,1) #return BxNxD

def get_raw_model_and_loss(N,num_layers=5,inner_layer_dim=64): 
    #given a polymer size N, returns the network to be trained as well as the loss function for that polymer shape

    assert N%2 == 0,"N must be even"
    dof = 3*(N-2)
    prior = nf.distributions.DiagGaussian(dof) #prior is just Gaussian for each mode
    flow_layers = []
    for i in range(num_layers):
        param_map = nf.nets.MLP([dof//2, inner_layer_dim, inner_layer_dim, dof], init_zeros=True)
        flow_layers.append(nf.flows.AffineCouplingBlock(param_map,split_mode="checkerboard"))
    model = nf.NormalizingFlow(prior,flow_layers)

    loss_f = energyfunc_loss(N)
    return model,loss_f

def run(N = 1000,ifsave=True,nam='model',use_pretrained=False,load_nam='model'):
    
    if use_pretrained:
        model = torch.load('models/'+load_nam+'.pickle')
        loss_func = energyfunc_loss(N)
    else:
        model,loss_func = get_raw_model_and_loss(N,num_layers=10)
    
    model = model.to(torch_device)
    loss_func = loss_func.to(torch_device)

    max_iter = 1000
    show_iter = 100
    num_samples = 1000
    #clip_value = 5 #using gradient clipping
    loss_hist = np.array([])
    
    torch.autograd.set_detect_anomaly(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    param_hist = []
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
    #        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
        
        loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())

    if ifsave:
        torch.save(model,'models/'+nam+'.pickle')
        
        # Plot loss
        plt.figure(figsize=(10, 10))
        plt.plot(loss_hist, label='loss')
        plt.legend()
        plt.savefig('/gpfs/commons/home/ieshghi/public_html/polygen/loss_hist.png')
    return loss_hist,model,param_hist
