import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import cKDTree
import poly_points as pl
import torch

sns.set_theme()

def draw_polymer(model,nam='plot'):
    plt.close('all')
    n_plot = 3 
    n_data = 100
    conf = model(model.q0.sample(n_data))
    polys,steps = pl.randtopoly(conf)
    steplengths = np.linalg.norm(steps.cpu().detach().numpy(),axis = 2)

    poly_len = polys.shape[1]

    loss_f = pl.energyfunc_loss(poly_len)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    cmap = plt.get_cmap('plasma')
    colors = [cmap(i) for i in np.linspace(0, 1, n_plot)]
    for i in range(n_plot):
        poly = polys[i].cpu().detach().numpy()
        color = colors[i]
        ax.plot(poly[:,0],poly[:,1],poly[:,2],'-',color=color,alpha=0.7)
    plt.savefig('/gpfs/commons/home/ieshghi/public_html/polygen/'+nam+'.png')
    plt.close('all')

    msds = msd(polys)
    nv = np.arange(0,polys.shape[1]-1)

    fig = plt.figure(figsize=(10,18))
    ax1 = fig.add_subplot(3,1,1)
    ax1.loglog(nv,msds,'.',label='Simulated chains')
    ax1.loglog(nv,0.1*nv**(2/3),'--',label='Fracal Globule scaling')
    ax1.set_xlabel('s')
    ax1.set_ylabel('MSD')
    ax1.legend()
    ax2 = fig.add_subplot(3,1,2)
    sns.histplot(steplengths.flatten())
    ax2.set_xlabel('Bond lengths')
    ax3 = fig.add_subplot(3,1,3)
    svals = calculate_contacts(polys,cutoff=0.8)    
    ax3.set_xlabel('s')
    log_s = np.log10(svals)
    sns.kdeplot(svals,log_scale=(10,10),clip=(1,100))
    xv = np.linspace(10,100,2)
    ax3.plot(xv,5*xv**(-1),'--',label='fractal scaling')
    ax3.legend()
#    ax3.plot(bins,bins**(-1)*10**5)
#    ax3.set_yscale('log')
#    ax3.set_xscale('log')

    fig.tight_layout()
    plt.savefig('/gpfs/commons/home/ieshghi/public_html/polygen/'+nam+'_msd.png')

def calculate_contacts(data, cutoff=0.8):
    #stolen from mirnylab's Polychrom package
    """Calculates contacts between points given the contact radius (cutoff) 

    Parameters
    ----------
    data : Nx3 array
        Coordinates of points
    cutoff : float , optional
        Cutoff distance (contact radius)

    Returns
    -------
    k by 2 array of contacts. Each row corresponds to a contact.
    """
    svals = np.zeros(0)
    for i in range(data.shape[0]):
        tree = cKDTree(data[i].cpu().detach().numpy())
        pairs = tree.query_pairs(cutoff, output_type="ndarray")
        svals = np.append(svals,np.diff(pairs,axis=1).flatten())

    return svals

def msd(xyz):
    nb = xyz.shape[0]
    n = xyz.shape[1]
    shifts = np.arange(0,n-1)
    msds = np.zeros(shifts.size)
    xyz = xyz.cpu().detach().numpy()

    for i, shift in enumerate(shifts):
        diffs = xyz[:,:-shift if shift else None,:] - xyz[:,shift:,:]
        msds[i] = np.square(diffs).mean()

    return msds
