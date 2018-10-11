import numpy as np

def genpoly(n,d):
    poly = np.zeros((n,d))
    nbours = np.zeros((2*d,d))
    for i in range(d):
        nbours[i,i] = 1
        nbours[i+d,i] = -1
    keep = np.zeros((1,d))
    for i in range(1,n):
        x = poly[i-1,:]
        poss = nbours + x
        for j in range(2*d):
            if(checkocc(poly,i,poss[j,:])==True):
                keep = vstack((keep,poss[j,:]))
        keep = keep[1:,:]        
        pick = np.random.choice(range(keep.shape(1))) #the only random step in the process
        pickpos = keep[pick,:]
        poly[i,:] = pickpos
    return poly    

def checkocc(poly,i,test):
    d = poly.shape[1]
    cons = poly[:i,:]
    truth = (sum(cons==test,1)==d)
    if sum(truth)==0:
        return True
    else:
        return False

def calcr(n,d,nsamp):
    rs = zeros((nsamp))
    for i in range(nsamp):
        poly = genpoly(n,d)
        r = np.sqrt(sum((poly[0,:]-poly[-1,:])**2))
        rs[i] = r
    return np.mean(rs),np.std(rs)
