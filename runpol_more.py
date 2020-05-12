import simpol as sp
from numpy import *
import sys

typ = int(sys.argv[1])
n = int(sys.argv[2])

lmax = 10

names = array(["a","b","c","d","e","f","g","h","i","j"])
lams = linspace(0,lmax,10)

l = 1
k = 10
g = 10
m = 0.2
T = 1

nsamp = 1
nstep = 1#10**5
pollength = 30

mr,sr,rv = sp.getscal(nsamp,pollength,l,k,g,m,T,lams[typ],0.01,nstep,ifid=0)
savetxt("datfiles/"+names[typ]+str(n)+".txt",rv)
