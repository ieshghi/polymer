import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from numpy import *
from numpy.random import rand
from numpy.random import randn

# Simulating polymer dynamics with a variable virial coefficient: can we see swelling and collapse?

# Global variables
alpha = 3.1730728678
beta = -0.856228645
tpow = 1.122462048309373

def backbone_evol(p0,l,k,T,lam,dt,nsteps,ifmov=0):
    if ifmov:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
    
    if dt == 0:
        b = 0
    else:
        b = sqrt(2*T/dt)

    np = p0.shape[1]
    print(b*dt)
    print(((2**(1/6)-1)*l*0.6))
    noises = b*randn(3,np,nsteps)
    rsqhist = zeros(nsteps)
    pos = p0.copy()
    pold = pos + b*randn(3,np)
    for i in range(nsteps):

        # for plotting and background info
        pc = mean(pos,1)
        pplot = pos-array([pc,]*(np)).T
        rsqhist[i] = rg(pplot)
        if ifmov:
            ax.plot3D(xs = pplot[0,:],ys = pplot[1,:],zs = pplot[2,:],marker=".")
            fig.savefig('/home/ieshghi/Documents/NYU/entropy_bio/project/picdump/fr'+str(i)+'.jpg')
            ax.clear()
        ###

        ### Actual time evolution
        force = spring(pos,l,k)+manybod(pos,lam,l,T)
        dp = timestep(pos,pold,force,noises[:,:,i],dt)
        pold = pos.copy()
        pos += dp

    if ifmov:
        plt.close('all')

    return pos,rsqhist

def timestep(pos,pold,force,noise,dt):
    return dt*(force+noise)
    ### Verlet
    #return pos-pold+(dt**2)*(force+noise)


def manybod(pos,lam,l,T):
    np = pos.shape[1]
    sig = l*0.3
    eps = T
    mbod = zeros(pos.shape)
    for i in range(np):
        mbod[:,i] = monomer_force(i,pos,eps,lam,sig)
    
    return mbod
    
def monomer_force(i,pos,eps,lam,sig):
    np = pos.shape[1]
    ri = pos[:,i]
    di = pos-array([ri,]*(np)).T
    fi = zeros(di.shape)
    for j in range(np):
        fi[:,j] = finterac(di[:,j],eps,lam,sig)
        
    return sum(fi,1)

def finterac(r,eps,lam,sig):
    rmin = tpow*sig
    p = sqrt(r.dot(r))
    if (p<rmin)and(p>0):
        rv = r/p
        return rv*(4*eps*(12*(sig**12)/(p**13)-6*(sig**6)/(p**7)))
    elif (p<(1.5*sig))and(p>0):
        rv = r/p
        return rv*(lam*p*alpha*eps*sin(alpha*p**2+beta))
    else:
        return 0*r

def finterac_sc(p,eps,lam,sig):
    rmin = tpow*sig
    if (p<rmin)and(p>0):
        return (4*eps*(12*(sig**12)/(p**13)-6*(sig**6)/(p**7)))
    elif (p<(1.5*sig))and(p>0):
        return (lam*p*alpha*eps*sin(alpha*p**2+beta))
    else:
        return 0

def interac(r,eps,lam,sig):
    rmin = tpow*sig
    if r<rmin:
        return 4*eps*((sig/r)**12-(sig/r)**6)+eps
    elif r<(1.5*sig):
        return eps*(-0.5+0.5*cos(alpha*r**2+beta))
    else:
        return 0

def rsq_scaling(lam):
    nl = 10
    ls = logspace(1,2.5,nl)
    nsteps = 10^5
    rsqs = zeros(nl)
    rsqsp = zeros(nl)
    for i in range(nl):
        p0 = zeros((3,int(floor(ls[i]))))
        p0[2,:] = linspace(-ls[i]/2,ls[i]/2,ls[i])
        pos,rsqh = backbone_evol(p0,1,1,1,lam,0.01,nsteps,0)
        rsqs[i] = mean(rsqh)
        rsqsp[i] = std(rsqh)

    return rsqs,rsqsp

def sample_walk(n):
    r1 = rand(n)
    r2 = rand(n)
    one = ones(n)
    d3 = 2*r1-one
    th = 2*pi*r2
    r = sqrt(one-(d3**2))
    d1 = r*cos(th)
    d2 = r*sin(th)
    xw = cumsum(d1)
    yw = cumsum(d2)
    zw = cumsum(d3)

    return vstack((xw,yw,zw))
	
def rg(pos):
    n = pos.shape[1]
    return sqrt(sum(pos**2)/n)

def spring(pos,l,k):
    d = (roll(pos,[0,-1])-pos)[:,:-1]
    n = shape(pos)[1]
    spr = zeros(pos.shape)
    for i in range(n):
        if i > 0 and i < n-1:
            spr[:,i] = -harm(d[:,i],l,k)+harm(d[:,i-1],l,k)
        elif i==1:
            spr[:,i] = -harm(d[:,i],l,k)
        else:
            spr[:,i] = harm(d[:,i-1],l,k)
    return spr

def harm(r,l,k):
    return -k*(r-l*r/sqrt(r.dot(r)))
