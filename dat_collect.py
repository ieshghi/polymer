import sys
from numpy import *
n_each = int(sys.argv[1])
ex = loadtxt("datfiles/ideal1.txt")

#first, collect ideals
r_id = zeros((n_each,ex.size))

for i in range(n_each):
    r_id[i,:] = loadtxt("datfiles/ideal"+str(i+1)+".txt")

savetxt("datfiles/ideals.txt",r_id)

#then, collect swollen
r_swol = zeros((n_each,ex.size))

for i in range(n_each):
    r_swol[i,:] = loadtxt("datfiles/swol"+str(i+1)+".txt")

savetxt("datfiles/swols.txt",r_swol)

#first, collect ideals
r_coll = zeros((n_each,ex.size))

for i in range(n_each):
    r_coll[i,:] = loadtxt("datfiles/coll"+str(i+1)+".txt")

savetxt("datfiles/colls.txt",r_coll)
