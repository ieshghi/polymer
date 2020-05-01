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

def bigjob():
    l = 1
    k = 10
    g = 10
    m = 0.2
    T = 1
    lam_swol = 0
    lam_coll = 10

    nsamp = 100
    nstep = 10**5
    pollength = 30

    print("getting ideal scaling")
    mr_id,sr_id,rv_id = getscal(nsamp,pollength,l,k,g,m,T,0,0.01,nstep,1)

    print("\n getting swollen scaling")
    mr_swol,sr_swol,rv_swol = getscal(nsamp,pollength,l,k,g,m,T,lam_swol,0.01,nstep,0)

    print("\n getting collapsed scaling")
    mr_coll,sr_coll,rv_coll = getscal(nsamp,pollength,l,k,g,m,T,lam_coll,0.01,nstep,0)

    print("\n saving data...")
    savetxt("mr_id.txt",mr_id)
    savetxt("sr_id.txt",sr_id)
    savetxt("rv_id.txt",rv_id)
    savetxt("mr_swol.txt",mr_swol)
    savetxt("sr_swol.txt",sr_swol)
    savetxt("rv_swol.txt",rv_swol)
    savetxt("mr_coll.txt",mr_coll)
    savetxt("sr_coll.txt",sr_coll)
    savetxt("rv_coll.txt",rv_coll)
    
def getscal(nsamp,npart,l,k,g,m,T,lam,dt,nsteps,ifid=0):
    p0 = sample_walk(npart)
    rvals = zeros((nsamp,npart))
    for i in range(nsamp):
        pos,d = backbone_evol(p0,l,k,g,m,T,lam,dt,nsteps,0,ifid)
        n,r = msd(pos)
        rvals[i,:] = r
    
    return mean(rvals,0),std(rvals,0),rvals

def backbone_evol(p0,l,k,g,m,T,lam,dt,nsteps,ifmov=0,ifid = 0):
    sig = 0.6*l
    if (ifid!=0)and(ifid!=1):
        ifid = 0

    if ifmov:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
    
    mag = sqrt(2*g*T*dt)

    np = p0.shape[1]
    noises = mag*randn(3,np,nsteps+1)
    rsqhist = zeros(nsteps)
    dhist = zeros((nsteps,np-1))
    pos = p0.copy()
    pold = pos + mag*randn(3,np)
    for i in range(nsteps):
        # for plotting and background info
        pc = mean(pos,1)
        pplot = pos-array([pc,]*(np)).T
        dists = diff(pos)
        dhist[i,:] = sqrt(sum(dists**2,0))
        
        if ifmov:
            ax.set_xlim([-1,1])
            ax.set_ylim([-1,1])
            ax.set_zlim([-1,1])
            ax.plot3D(xs = pplot[0,:],ys = pplot[1,:],zs = pplot[2,:],marker=".")
            fig.savefig('/home/ieshghi/Documents/NYU/entropy_bio/polymer/picdump/fr'+str(i)+'.jpg')
            ax.clear()
        ###

        ### Actual time evolution

        a = (1-g*dt/(2*m))/(1+g*dt/(2*m))
        b = 1/(1+g*dt/(2*m))
        force = spring(pos,l,k)+(1-ifid)*manybod(pos,lam,sig,T)


        dp = checksize(timestep_verlet(pos,pold,force,noises[:,:,i],noises[:,:,i+1],dt,a,b,m),l,sig)
        pold = pos.copy()
        pos += dp

    if ifmov:
        plt.close('all')

    return pos,dhist

def checksize(jump,l,sig):
    lengths = sqrt(sum(jump**2,0))
    kill = lengths>l
    jump[:,kill] = (jump[:,kill]/lengths[kill])*sig
    return jump

def timestep_verlet(pos,pold,force,noise,noise_fut,dt,a,b,m):
    return pos*(2*b-1)-a*pold+b*dt**2/m*force+b*dt*(noise+noise_fut)/(2*m)

def manybod(pos,lam,sig,T):
    np = pos.shape[1]
    eps = T
    mbod = zeros(pos.shape)
    for i in range(np):
        mbod[:,i] = -monomer_force(i,pos,eps,lam,sig) 
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
    p = sqrt(r.dot(r))
    a = p/sig
    rmin = tpow
    if p == 0:
        return 0
    elif (a<rmin):
        rv = r/p
        return rv*(4*eps*(12*(1/a)**13-6*(1/a)**7)/p)
    elif (a<(1.5)):
        rv = r/p
        return rv*lam*a*alpha*eps*sin(alpha*a**2+beta)/p
    else:
        return 0*r

def finterac_sc(p,eps,lam,sig):
    a = p/sig
    rmin = tpow
    if p==0:
        return 0
    elif (a<rmin):
        return 4*eps*(12*(1/a)**13-6*(1/a)**7)/p
    elif (a<1.5):
        return lam*a*alpha*eps*sin(alpha*a**2+beta)/p

   # if (p<rmin)and(p>0):
   #     return (4*eps*(12*(sig**12)/(p**13)-6*(sig**6)/(p**7)))
   # elif (p<(1.5*sig))and(p>0):
   #     return (lam*p*alpha*eps*sin(alpha*p**2+beta))
   # else:
   #     return 0

def interac(r,eps,lam,sig):
    rmin = tpow*sig
    if r<rmin:
        return 4*eps*((sig/r)**12-(sig/r)**6)+eps*(1-lam)
    elif r<(1.5*sig):
        return lam*eps*(-0.5+0.5*cos(alpha*r**2+beta))
    else:
        return 0

def rsq_scaling(lam):
    nl = 5
    ls = logspace(1,2.5,nl)
    nsteps = 10**4
    rsqs = zeros(nl)
    rsqsp = zeros(nl)
    for i in range(nl):
        p0 = sample_walk(int(ls[i]))
        pos,rsqh = backbone_evol(p0,1,1,1,lam,0.00001,nsteps,0)
        rsqs[i] = mean(rsqh)
        rsqsp[i] = std(rsqh)

    return ls,rsqs,rsqsp

def msd(pos):
    np = pos.shape[1]
    pc = pos[:,0]
    pos = pos-array([pc,]*(np)).T
    r = sqrt(sum(pos**2,0))
    n = linspace(0,r.size,r.size)
    return n,r

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
    d = diff(pos)
    n = shape(pos)[1]
    spr = zeros(pos.shape)
    for i in range(n):
        if i > 0 and i < n-1:
            spr[:,i] = harm(d[:,i],l,k)-harm(d[:,i-1],l,k)
        elif i==1:
            spr[:,i] = harm(d[:,i],l,k)
        elif i==n-1:
            spr[:,i] = -harm(d[:,i-1],l,k)
    return spr

def harm(r,l,k):
    a = sqrt(r.dot(r))
    return k*(a-l)*r/a
