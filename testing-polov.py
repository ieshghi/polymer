import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.func import jacrev 
from tqdm import tqdm

def get_svals(N):
    inds = torch.arange(N)
    smat = torch.zeros(N,N)
    for i in range(N):
        smat[i,:] = torch.abs(inds-i)
    return smat*1.0

def smoothstep(x,k=10): #logistic function with a given steepness
    return (1 - 1/(1+torch.exp(-k*(x-1))))


N = 1000
nsteps = 1e4
tau = 0.1

k = 1
sigma = 1
lcol = 0.1
smat = get_svals(N)

def energy(conf):

    dists = torch.cdist(conf).triu(diagonal=1) #conf should be of shape (N,3)
    gamma = -8/3
    spring_e = (smat**gamma)*dists**2

    batchmean_collisions = torch.mean(smoothstep(dists[dists>0]/self.lcol),axis=0) #punish collisions with a smoothed step function
    collision_energy = sigma*torch.sum(batchmean_collisions)

    return collision_energy + spring_e

forcefunc = jacrev(energy)(conf)

def qfunc(xprime,x):
    u = torch.sum((xprime-x-tau*forcefunc(x))**2)
    return torch.exp(-u/(4*tau))

def alpha(xnew,xold):
    numer = energy(xnew)*qfunc(xold,xnew)
    denom = energy(xold)*qfunc(xnew,xold)
    return torch.min(1,numer/denom)

x = torch.randn(N,3)
noises = torch.randn(nsteps,N,3)
i = 0
for it in tqdm(range(nsteps)):
    force = -forcefunc(x)
    noise = noises[i]

    xhat = x + tau*force + torch.sqrt(2*tau)*noise
    u = torch.rand()
    if u <= alpha(xhat,x):
        x = xhat

    i += 1
