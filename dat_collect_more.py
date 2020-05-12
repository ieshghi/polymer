import sys
from numpy import *
n_each = int(sys.argv[1])
ex = loadtxt("datfiles/ideal1.txt")

names = array(["a","b","c","d","e","f","g","h","i","j"])

r_id = zeros((n_each,ex.size))

for j in range(10):
    for i in range(n_each):
        r_id[i,:] = loadtxt("datfiles/"+names[j]+str(i+1)+".txt")
    savetxt("datfiles/"+names[j]+"s.txt",r_id)
