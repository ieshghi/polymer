import numpy as np
import matplotlib.pyplot as plt

def genpoly(n,d):
    poly = np.zeros((n,d))
    nbours = np.zeros((2*d,d))
    for i in range(d):
        nbours[i,i] = 1
        nbours[i+d,i] = -1
    for i in range(1,n):
        keep = np.zeros((1,d))
        x = poly[i-1,:]
        poss = nbours + x
        for j in range(2*d):
            if(checkocc(poly,i,poss[j,:])==True):
                keep = np.vstack((keep,poss[j,:]))
        keep = keep[1:,:]        
        if keep.size == 0:
            return np.nan
            break
        pick = np.random.choice(range(keep.shape[0])) #the only random step in the process
        pickpos = keep[pick,:]
        poly[i,:] = pickpos
    return poly 

            
def checkocc(poly,i,test):
    d = poly.shape[1]
    cons = poly[:i-1,:]
    truth = (sum(cons==test,1)==d)
    if np.size(truth)==1:
        return True
    else:
        if sum(truth)==0:
            return True
        else:
            return False

def calcr(nin,d,nsamp):
    n = int(np.ceil(nin))
    rs = np.zeros((nsamp))
    for i in range(nsamp):
        poly = np.nan
        while np.isnan(np.sum(poly)):
            poly = genpoly(n,d)

        r = np.sqrt(sum((poly[0,:]-poly[-1,:])**2))
        rs[i] = r
    return np.mean(rs),np.std(rs)

def plotpoly(poly):
    if poly.shape[1]!=2:
        return 1
    else:
       plt.plot(poly[:,0],poly[:,1])
