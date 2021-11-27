#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 17:43:38 2021

@author: amartinez
"""

import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import dynesty
from astropy.stats import sigma_clip
from astropy.stats import sigma_clipped_stats
from matplotlib.ticker import FormatStrFormatter
import scipy.integrate as integrate
from scipy import stats
from scipy.stats import norm
# In[3]:
band='H'
folder='im_jitter_NOgains/'
exptime=10
data='/Users/amartinez/Desktop/PhD/python/Gaussian_fit/'
tmp='/Users/amartinez/Desktop/PhD/HAWK/The_Brick/photometry/054_'+band+'/dit_'+str(exptime)+'/'+folder+'tmp_bs/'
pruebas='/Users/amartinez/Desktop/PhD/HAWK/The_Brick/photometry/pruebas/'
# In[4]:


# plt.rcParams['figure.figsize'] = (20,10)
from matplotlib import rcParams
rcParams.update({'xtick.major.pad': '7.0'})
rcParams.update({'xtick.major.size': '7.5'})
rcParams.update({'xtick.major.width': '1.5'})
rcParams.update({'xtick.minor.pad': '7.0'})
rcParams.update({'xtick.minor.size': '3.5'})
rcParams.update({'xtick.minor.width': '1.0'})
rcParams.update({'ytick.major.pad': '7.0'})
rcParams.update({'ytick.major.size': '7.5'})
rcParams.update({'ytick.major.width': '1.5'})
rcParams.update({'ytick.minor.pad': '7.0'})
rcParams.update({'ytick.minor.size': '3.5'})
rcParams.update({'ytick.minor.width': '1.0'})
rcParams.update({'font.size': 20})
rcParams.update({'figure.figsize':(10,5)})
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = 'dejavuserif'
plt.rcParams['text.usetex'] = False
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})

#%%
# step=np.arange(1.0,1.75,0.25)#these have worked
# auto='auto'
auto='auto'
if auto =='auto':
    step=np.arange(0,2,1)#
else:
    step=np.arange(1.5,1.6,0.1)#also works if running each bing width one by one, for some reason...

media_amp=[]
print(step)

for sloop in range(len(step)-1):
    chip='both'
    
    # if auto != 'auto':
    #     list_bin=np.arange(-15,15,step[sloop])
    #     auto=list_bin
    #     print(list_bin)
    #     nbins=len(list_bin)-1
    #     print(30*'#'+'\n'+'nbins=%s'%(nbins)+'\n'+30*'#')

    # nbins=9
    accu=1.5
    sm=0.5
    in_brick=1#slect list in or out brick
    
    if in_brick==1:
        if chip =='both':
            v_x2,v_y2,dvx2,dvy2,mh2,m2,ar,dec,arg,decg=np.loadtxt(data+'DOWN_aa_IDL_arcsec_vx_vy_chip2.txt',unpack=True)
            v_x3,v_y3,dvx3,dvy3,mh3,m3,ar,dec,arg,decg=np.loadtxt(data+'UP_aa_IDL_arcsec_vx_vy_chip3.txt',unpack=True)
            v_x=np.r_[v_x2,v_x3]
            v_y=np.r_[v_y2,v_y3]
            dvx=np.r_[dvx2,dvx3]
            dvy=np.r_[dvy2,dvy3]
            mh=np.r_[mh2,mh3]
            m=np.r_[m2,m3]
            lst=np.loadtxt(tmp+'aa_IDL_lst_chip3.txt')
            print(30*'#'+'\n'+'lst=%s'%(lst)+'\n'+30*'#')
        elif chip==2 or chip==3:
            # lst=np.loadtxt(tmp+'aa_IDL_lst_chip%s.txt'%(chip))
            lst=np.loadtxt(tmp+'aa_IDL_lst_chip%s.txt'%(chip))
            print(30*'#'+'\n'+'lst=%s'%(lst)+'\n'+30*'#')
            # v_x,v_y,dvx,dvy=np.loadtxt(data+'arcsec_vx_vy_chip%s.txt'%(chip),unpack=True)
            # v_x,v_y,dvx,dvy,mh,m=np.loadtxt(data+'IDL_arcsec_vx_vy_chip%s.txt'%(chip),unpack=True)
            #add 'aa' in front of the list name to used aa aligned lists
            if chip==3:
                v_x,v_y,dvx,dvy,mh,m,ar,dec,arg,decg=np.loadtxt(data+'UP_aa_IDL_arcsec_vx_vy_chip%s.txt'%(chip),unpack=True)
            if chip==2:
                v_x,v_y,dvx,dvy,mh,m,ar,dec,arg,decg=np.loadtxt(data+'DOWN_aa_IDL_arcsec_vx_vy_chip%s.txt'%(chip),unpack=True)

            # v_x,v_y,dvx,dvy,mh,m=np.loadtxt(data+'IDL_arcsec_vx_vy_chip%s.txt'%(chip),unpack=True)
            
    mh_all=mh
    m_all=m
    dvx_all=dvx
    dvy_all=dvy
    
    sel_m=np.where(abs(mh-m)<sm)
    v_x=v_x[sel_m]
    v_y=v_y[sel_m]
    mh=mh[sel_m]
    m=m[sel_m]
    dvx=dvx[sel_m]
    dvy=dvy[sel_m]
    
    sel=np.where((dvx<accu)&(dvy<accu))
    v_x=v_x[sel]
    v_y=v_y[sel]
    mh=mh[sel]
    fig,ax=plt.subplots(1,1)
    sig_h=sigma_clip(v_y,sigma=5,maxiters=20,cenfunc='mean',masked=True)
    v_y=v_y[sig_h.mask==False]
    # h=ax.hist(v_y,bins=list_bin,edgecolor='black',linewidth=2,density=True)
    if auto != 'auto':
        list_bin=np.arange(min(v_y),max(v_y),step[sloop])
        auto=list_bin
        print(list_bin)
        nbins=len(list_bin)-1
        print(30*'#'+'\n'+'nbins=%s'%(nbins)+'\n'+30*'#')
    
    h=ax.hist(v_y,bins=auto,edgecolor='black',linewidth=2,density=True)
    h1=np.histogram(v_y,bins=auto,density=False)

    
    print(35*'-'+'\n'+'The width of the bin is: %.3f'%(h[1][3]-h[1][2])+'\n'+35*'-')
    x=[h[1][i]+(h[1][1]-h[1][0])/2 for i in range(len(h[0]))]#middle value for each bin
    ax.axvline(np.mean(v_y), color='r', linestyle='dashed', linewidth=3)
    ax.legend(['Chip=%s, %s, mean= %.4f, std=%.2f'
                  %(chip,len(v_y),np.mean(v_y),np.std(v_y))],fontsize=12,markerscale=0,shadow=True,loc=1,handlelength=-0.0)
    y=h[0]#height for each bin
    w=h[1]
    pij=np.empty([len(v_y),len(h[0])])
    for b in range(len(y)):
        for v in range(len(v_y)):
            # snd = stats.norm(v_y[v],dvy[v])
            pij[v,b]=norm(v_y[v],dvy[v]).cdf(w[b+1])-norm(v_y[v],dvy[v]).cdf(w[b])
    vj = [np.sum(pij[:,j]*(1 - pij[:,j])) for j in range(len(h1[1])-1)]
    sj=np.sqrt(vj)   
    sj_n=sj/(len(v_y)*(h[1][1]-h[1][0]))
    yerr=sj_n
    vy_p=[np.sum(pij[:,i]) for i in range(len(h1[1])-1)]
    fig,ax=plt.subplots(1,1)
    ax.scatter(x,vy_p/(len(v_y)*(h[1][1]-h[1][0])),alpha=0.3,color='red')
    ax.hist(v_y,bins=auto,edgecolor='black',linewidth=2,density=True,alpha=0.5)
    ax.errorbar(x,vy_p/(len(v_y)*(h[1][1]-h[1][0])),sj/(len(v_y)*(h[1][1]-h[1][0])))
    
    
    #%%
    vy_p=[np.sum(pij[:,i]) for i in range(len(h1[1])-1)]
    fig,ax=plt.subplots(1,1)
    ax.scatter(x,vy_p/(len(v_y)*(h[1][1]-h[1][0])),alpha=0.3,color='red')
    ax.hist(v_y,bins=auto,edgecolor='black',linewidth=2,density=True,alpha=0.5)
    ax.errorbar(x,vy_p/(len(v_y)*(h[1][1]-h[1][0])),sj/(len(v_y)*(h[1][1]-h[1][0])))
    
    np.savetxt(pruebas+'vy_in_pij_accu%s_sm%s.txt'%(accu,sm),pij)
    # pij=np.loadtxt(pruebas+'vx_in_pij_accu%s_sm%s.txt'%(accu,sm))
#%%
    y= vy_p/(len(v_y)*(h[1][1]-h[1][0]))   
    
#%%   
    def gaussian(x, mu, sig, amp):
        return amp * (1 / (sig * (np.sqrt(2 * np.pi)))) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    
    
    # In[8]:
    
    
    def loglike(theta):
        mu1, sigma1, amp1,mu2,sigma2,amp2 = theta
        model = gaussian(x, mu1, sigma1, amp1)+gaussian(x,mu2,sigma2,amp2)
     
        return -0.5 * np.sum(((y - model)/yerr) ** 2)#chi squared model
    
    
    def prior_transform(utheta):
        """Transforms the uniform random variable `u ~ Unif[0., 1.)`
        to the parameter of interest `x ~ Unif[-10., 10.)`."""
        #x = 2. * u - 1.  # scale and shift to [-1., 1.)
        #x *= 10.  # scale to [-10., 10.)
        umu1, usigma1, uamp1,  umu2, usigma2, uamp2= utheta
    
    #     mu1 = -1. * umu1-8   # scale and shift to [-10., 10.)
        mu1 =1*umu1-0.5 # scale and shift to [-3., 3.)
        sigma1 = (usigma1)*3
        amp1 = uamp1    
    
        
        # mu2 = -0.018+ (0.062*umu2-0.031)# scale and shift to [-3., 3.)
        # sigma2 = 2.9+(0.15*usigma2-0.075)
        mu2= 2*umu2-1
        sigma2=6*usigma2
        amp2 = uamp2
    
        return mu1, sigma1, amp1, mu2, sigma2, amp2
    # prior transform
    # def prior_transform(utheta):
    #     um, ub, ulf = utheta
    #     m = 5.5 * um - 5. #### [0, 5.5] and then [-5, 0.5]
    #     mu, sigma=5, 3
    #     b  = stats.norm.ppf(ub, loc=mu, scale=sigma)
    #     lnf = 11. * ulf - 10.
        
    #     return m, b, lnf
    
    
    # In[9]:
    
    
    sampler = dynesty.NestedSampler(loglike, prior_transform, ndim=6, nlive=200,
                                            bound='multi', sample='rwalk')
    sampler.run_nested()
    res = sampler.results
    
    
    # In[10]:
    
    
    from dynesty import plotting as dyplot
    rcParams.update({'xtick.major.pad': '7.0'})
    rcParams.update({'xtick.major.size': '7.5'})
    rcParams.update({'xtick.major.width': '1.5'})
    rcParams.update({'xtick.minor.pad': '7.0'})
    rcParams.update({'xtick.minor.size': '3.5'})
    rcParams.update({'xtick.minor.width': '1.0'})
    rcParams.update({'ytick.major.pad': '7.0'})
    rcParams.update({'ytick.major.size': '7.5'})
    rcParams.update({'ytick.major.width': '1.5'})
    rcParams.update({'ytick.minor.pad': '7.0'})
    rcParams.update({'ytick.minor.size': '3.5'})
    rcParams.update({'ytick.minor.width': '1.0'})
    rcParams.update({'font.size': 20})
    
    
    # truths = [mu1_true, sigma1_true, amp1_true, mu2_true, sigma2_true, amp2_true]
    labels = [r'$mu1$', r'$sigma1$', r'$amp1$', r'$mu2$', r'$sigma2$', r'$amp2$']
    # fig, axes = dyplot.traceplot(sampler.results, truths=truths, labels=labels,
    #                              fig=plt.subplots(6, 2, figsize=(16, 27)))
    
    fig, axes = dyplot.traceplot(sampler.results,labels=labels,
                                 fig=plt.subplots(6, 2, figsize=(16, 20)))
    plt.show()
    
    
    # In[11]:
    
    
    # fig, axes = dyplot.cornerplot(res, truths=truths, color='blue', show_titles=True, 
    #                               title_kwargs={'x': 0.65, 'y': 1.05}, labels=labels,
    # 
    # %%                              fig=plt.subplots(6, 6, figsize=(28, 28)))
    # This is de corner plot
    fig, axes = dyplot.cornerplot(res, color='blue', show_titles=True, 
                                  title_kwargs={'x': 0.65, 'y': 1.05}, labels=labels,
                                  fig=plt.subplots(6, 6, figsize=(28, 28)))
    
    
    plt.show() 
    
    
    # In[12]:
    
    
    res.summary()
    
    
    # In[13]:
    
    
    from dynesty import utils as dyfunc
    
    samples, weights = res.samples, np.exp(res.logwt - res.logz[-1])
    mean, cov = dyfunc.mean_and_cov(samples, weights)
    print(mean)
    
    
    # In[14]:
    
    
    plt.figure(figsize =(8,8))
    # from matplotlib import rcParams
    # rcParams.update({'xtick.major.pad': '7.0'})
    # rcParams.update({'xtick.major.size': '7.5'})
    # rcParams.update({'xtick.major.width': '1.5'})
    # rcParams.update({'xtick.minor.pad': '7.0'})
    # rcParams.update({'xtick.minor.size': '3.5'})
    # rcParams.update({'xtick.minor.width': '1.0'})
    # rcParams.update({'ytick.major.pad': '7.0'})
    # rcParams.update({'ytick.major.size': '7.5'})
    # rcParams.update({'ytick.major.width': '1.5'})
    # rcParams.update({'ytick.minor.pad': '7.0'})
    # rcParams.update({'ytick.minor.size': '3.5'})
    # rcParams.update({'ytick.minor.width': '1.0'})
    # rcParams.update({'font.size': 20})
    # rcParams.update({'figure.figsize':(10,5)})
    # rcParams.update({
    #     "text.usetex": True,
    #     "font.family": "sans-serif",
    #     "font.sans-serif": ["Helvetica"]})
    
    
    rcParams.update({'font.size': 20})
    results = sampler.results
    print(results['logz'][-1])
    
    fig, ax = plt.subplots(figsize=(8,8))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    a=1#to chnge the axix a=-1
    # h=plt.hist(v_y*a, bins= auto, color='darkblue', alpha = 0.6, density =True, histtype = 'stepfilled')
    x=np.array(x)
    ax.scatter(x,vy_p/(len(v_y)*(h[1][1]-h[1][0])),alpha=0.3,color='red')
    ax.bar(x,vy_p/(len(v_y)*(h[1][1]-h[1][0])),width=h[1][1]-h[1][0],alpha=0.3,color='blue')
    xplot = np.linspace(min(x), max(x), 100)
    
    # plt.plot(xplot, gaussian(xplot, mean[0], mean[1], mean[2]) , color="darkorange", linewidth=3, alpha=0.6)
    
    plt.plot(xplot, gaussian(xplot*a, mean[0], mean[1], mean[2]) + gaussian(xplot*a, mean[3], mean[4], mean[5]), color="darkorange", linewidth=3, alpha=1)
    plt.plot(xplot, gaussian(xplot*a, mean[0], mean[1], mean[2])  , color="k", linestyle='dashed', linewidth=3, alpha=0.6)
    plt.plot(xplot, gaussian(xplot*a, mean[3], mean[4], mean[5])  , color="red", linestyle='dashed', linewidth=3, alpha=0.6)
    plt.xlim(-15,15)
    # plt.axvline(mean[0],linestyle='dashed',color='orange')
    # plt.axvline(mean[3],linestyle='dashed',color='orange')
    # plt.text(min(x),max(h[0]),'$\mu_{1}=%.3f$'%(mean[0]),color='k')
    # plt.text(min(x),max(h[0]-0.01),'$\sigma_{1}=%.3f$'%(mean[1]),color='k')
    # plt.text(min(x),max(h[0]-0.02),'$amp_{1}=%.3f$'%(mean[2]),color='k')
    # plt.text(max(x)/2,max(h[0]),'$\mu_{2}=%.3f$'%(mean[3]),color='red')
    plt.text(min(x),max(h[0]-0.05),'$logz=%.0f$'%(results['logz'][-1]),color='b')
    # # # if accu <10:
    # # #     plt.text(min(x),max(h[0]-0.05),'$\sigma_{vy}<%.1f\ mas\ a^{-1}$'%(accu),color='b')
    plt.text(min(x),max(h[0]-0.04),'nbins=%s'%(len(h[0])),color='b')
    # plt.text(max(x)/2,max(h[0]-0.01),'$\sigma_{2}=%.3f$'%(mean[4]),color='red')
    # plt.text(max(x)/2,max(h[0]-0.02),'$amp_{2}=%.3f$'%(mean[5]),color='red')
    # if (chip==2 or chip==3) and in_brick==1:
    #     plt.text(max(x)/2,max(h[0]-0.05),'$list = %.0f$'%(lst),color='b')
    # elif in_brick==0:
    #     if (chip==2 or chip==3):
    #         plt.text(max(x)/2,max(h[0]-0.05),'$list =%.0f %s$'%(lst,'out'),color='b')
    #     elif chip=='both':
    #         plt.text(max(x)/2,max(h[0]-0.05),'$list =%s %s$'%(lst,'out'),color='b')
    plt.ylabel('N')
    # # plt.xlabel(r'$\mu_{l}$ (Km s$^{-1}$)') 
    plt.xlabel(r'$\mathrm{\mu_{b} (mas\ a^{-1})}$') 
    plt.legend(['Zone A'],fontsize=20,markerscale=0,shadow=True,loc=2,handlelength=-0.0) 