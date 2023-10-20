import numpy as np
import matplotlib.pyplot as plt
import poly_points as pl
import torch

def draw_polymer(conf,nam='plot'):
    plt.close('all')
    polys,steps = pl.randtopoly(conf)
    steplengths = np.linalg.norm(steps.cpu().detach().numpy(),axis = 2)
    n_poly = polys.shape[0]
    ax = plt.figure().add_subplot(projection='3d')
    cmap = plt.get_cmap('plasma')
    colors = [cmap(i) for i in np.linspace(0, 1, n_poly)]
    for i in range(n_poly):
        poly = polys[i].cpu().detach().numpy()
        color = colors[i]
        ax.plot(poly[:,0],poly[:,1],poly[:,2],'-',color=color)
    plt.savefig('/gpfs/commons/home/ieshghi/public_html/polygen/'+nam+'.png')
    plt.close('all')

    msds = msd(polys)
    fig,(ax1,ax2) = plt.subplots(2,1)
    nv = np.arange(0,polys.shape[1]-1)
    ax1.loglog(nv,msds)
    ax1.loglog(nv,nv**(6/5),'--')
    ax1.loglog(nv,nv,'--')
    ax1.loglog(nv,nv**(2/3),'--')
    ax2.hist(steplengths.flatten(),100)
    plt.savefig('/gpfs/commons/home/ieshghi/public_html/polygen/'+nam+'_msd.png')

def xydist_point(conf,n=np.inf,nam='plot'):
    plt.close('all')
    loss_func = energyfunc_loss()
    n_poly = conf.shape[0]
    conf_list = [conf[i,:] for i in range(n_poly)]
    poly_list = list(map(loss_func.get_poly,conf_list))
    if np.isinf(n):
        n = poly_list[0].shape[0]-1
    for i in range(n_poly):
        poly = poly_list[i].detach().numpy()
        plt.plot(poly[n,0],poly[n,1],'k.')
    plt.savefig('/gpfs/commons/home/ieshghi/public_html/polygen/'+nam+'.png')

def conf_compare(conf1,conf2,nam = 'plot'):
    plt.close('all')
    loss_func = energyfunc_loss()
    n1 = conf1.shape[0]
    conf1_list = [conf1[i,:] for i in range(n1)]
    n2 = conf2.shape[0]
    conf2_list = [conf2[i,:] for i in range(n2)]
    poly1 = list(map(loss_func.get_poly,conf1_list))
    poly2 = list(map(loss_func.get_poly,conf2_list))
    msd1_list = list(map(msd,poly1))
    msd2_list = list(map(msd,poly2))
    n1_step = msd1_list[0].shape[0]
    n2_step = msd2_list[0].shape[0]
    t1 = np.arange(n1_step)
    t2 = np.arange(n2_step)
    msd1_mean = np.nanmean(msd1_list,axis=0)
    msd2_mean = np.nanmean(msd2_list,axis=0)
#    for i in range(n_samp):
#        plt.loglog(t,msd_list[i])
    plt.loglog(t1,msd1_mean,'-',label='Data 1')
    plt.loglog(t2,msd2_mean,'--',label='Data 2')
    plt.loglog(t1,np.ones(t1.shape)*n1_step**(1/3)*1.5,'k--')
    plt.legend()
    plt.savefig('/gpfs/commons/home/ieshghi/public_html/polygen/'+nam+'.png')
    return msd1_mean,msd2_mean

def conf_analyze(confdata,nam = 'plot'):
    plt.close('all')
    loss_func = energyfunc_loss()
    n_samp = confdata.shape[0]
    conf_list = [confdata[i,:] for i in range(n_samp)]
    poly_list = list(map(loss_func.get_poly,conf_list))
    msd_list = list(map(msd,poly_list))
    n_step = msd_list[0].shape[0]
    t = np.arange(n_step)
    msd_mean = np.mean(msd_list,axis=0)
#    for i in range(n_samp):
#        plt.loglog(t,msd_list[i])
    plt.loglog(t,msd_mean,'-',label='Data')
    plt.loglog(t,10*t,'--',label='Random Walk')
    plt.loglog(t,10*t**(6/5),'--',label='SAW')
    plt.legend()
    plt.savefig('/gpfs/commons/home/ieshghi/public_html/polygen/'+nam+'.png')
    return msd_list
    
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
