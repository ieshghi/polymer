import simpol as sp
from numpy import *
import sys

typ = int(sys.argv[1])
n = int(sys.argv[2])

l = 1
k = 10
g = 10
m = 0.2
T = 1
lam_swol = 0
lam_coll = 10

nsamp = 1
nstep = 10**5
pollength = 30

if typ==0: #ideal chain
    mr_id,sr_id,rv_id = sp.getscal(nsamp,pollength,l,k,g,m,T,0,0.01,nstep,1)
    savetxt("datfiles/ideal"+str(n)+".txt",rv_id)

elif typ==1: #swollen chain
    mr_swol,sr_swol,rv_swol = sp.getscal(nsamp,pollength,l,k,g,m,T,lam_swol,0.01,nstep,0)
    savetxt("datfiles/swol"+str(n)+".txt",rv_swol)

elif typ==2: #collapsed chain
    mr_coll,sr_coll,rv_coll = sp.getscal(nsamp,pollength,l,k,g,m,T,lam_coll,0.01,nstep,0)
    savetxt("datfiles/coll"+str(n)+".txt",rv_swol)
