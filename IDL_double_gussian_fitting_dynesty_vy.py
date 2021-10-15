#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Im going and try to follow this tutorial and see how it goes
#https://prappleizer.github.io/Tutorials/MCMC/MCMC_Tutorial.html


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import dynesty
from astropy.stats import sigma_clip
from astropy.stats import sigma_clipped_stats
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


# In[5]:
ran=1
for sloop in range(ran,ran+16):
    chip=3
    
    nbins=31-sloop
    
    accu=1.5
    sm=1
    in_brick=1#slect list in or out brick
    
    if in_brick==1:
        if chip =='both':
            v_x2,v_y2,dvx2,dvy2,mh2,m2=np.loadtxt(data+'aa_IDL_arcsec_vx_vy_chip2.txt',unpack=True)
            v_x3,v_y3,dvx3,dvy3,mh3,m3=np.loadtxt(data+'aa_IDL_arcsec_vx_vy_chip3.txt',unpack=True)
            v_x=np.r_[v_x2,v_x3]
            v_y=np.r_[v_y2,v_y3]
            dvx=np.r_[dvx2,dvx3]
            dvy=np.r_[dvy2,dvy3]
            mh=np.r_[mh2,mh3]
            m=np.r_[m2,m3]
        elif chip==2 or chip==3:
            lst=np.loadtxt(tmp+'IDL_lst_chip%s.txt'%(chip))
            # v_x,v_y,dvx,dvy=np.loadtxt(data+'arcsec_vx_vy_chip%s.txt'%(chip),unpack=True)
            # v_x,v_y,dvx,dvy,mh,m=np.loadtxt(data+'IDL_arcsec_vx_vy_chip%s.txt'%(chip),unpack=True)
            #add 'aa' in front of the list name to used aa aligned lists
            v_x,v_y,dvx,dvy,mh,m=np.loadtxt(data+'aa_IDL_arcsec_vx_vy_chip%s.txt'%(chip),unpack=True)
    elif in_brick==0:
        if chip=='both':
            lst='All '
            #something was weird with list 12 aligment, it didn't converg... 
            v_x10,v_y10,dvx10,dvy10,mh10,m10=np.loadtxt(data+'aa_IDL_arcsec_vx_vy_chip2_out_Brick10.txt',unpack=True)
            # v_x12,v_y12,dvx12,dvy12,mh12=np.loadtxt(data+'IDL_arcsec_vx_vy_chip3_out_Brick12.txt',unpack=True)
            v_x16,v_y16,dvx16,dvy16,mh16,m16=np.loadtxt(data+'aa_IDL_arcsec_vx_vy_chip3_out_Brick16.txt',unpack=True)
            
            # v_x=np.r_[v_x16,v_x12,v_x10]
            # v_y=np.r_[v_y16,v_y12,v_y10]
            # dvx=np.r_[dvx16,dvx12,dvx10]
            # dvy=np.r_[dvy16,dvy12,dvy10]
            # mh=np.r_[mh16,mh12,mh10]
            
            v_x=np.r_[v_x16,v_x10]
            v_y=np.r_[v_y16,v_y10]
            dvx=np.r_[dvx16,dvx10]
            dvy=np.r_[dvy16,dvy10]
            mh=np.r_[mh16,mh10]
            m=np.r_[m16,m10]
        else:
            lst=np.loadtxt(tmp+'IDL_lst_chip%s.txt'%(chip))
            v_x,v_y,dvx,dvy,mh,m=np.loadtxt(data+'aa_IDL_arcsec_vx_vy_chip%s_out_Brick%.0f.txt'%(chip,lst),unpack=True)        
    
    
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
    sig_h=sigma_clip(v_y,sigma=500,maxiters=20,cenfunc='mean',masked=True)
    v_y=v_y[sig_h.mask==False]
    h=ax.hist(v_y,bins=nbins,edgecolor='black',linewidth=2,density=True)
    x=[h[1][i]+(h[1][1]-h[1][0])/2 for i in range(len(h[0]))]#middle value for each bin
    ax.axvline(np.mean(v_y), color='r', linestyle='dashed', linewidth=3)
    ax.legend(['Chip=%s, %s, mean= %.4f, std=%.2f'
                  %(chip,len(v_y),np.mean(v_y),np.std(v_y))],fontsize=12,markerscale=0,shadow=True,loc=1,handlelength=-0.0)
    y=h[0]#height for each bin
    #yerr = y*0.05
    #yerr = y*0.01
    yerr=0.0001
    y += yerr
    ax.scatter(x,y,color='g',zorder=3)
    
    # In[5]:
    count=0
    for i in range(len(mh)):
        if abs(mh_all[i]-m_all[i])>sm:
            count+=1
    print(35*'#'+'\n'+'stars with diff in mag > %s: %s'%(sm,count)+'\n'+35*'#')
    
    # In[6]:
    
    ejes=[dvx_all,dvy_all]
    no_sel=np.where((dvx_all>accu)&(dvy_all>accu))
    no_m=np.where(abs(mh_all-m_all)>sm)
    ejes_accu=[dvx_all[no_sel],dvy_all[no_sel]]
    ejes_m=[dvx_all[no_m],dvy_all[no_m]]
    names=['x','y']
    if accu<50:
        fig, ax=plt.subplots(1,2,figsize=(20,10))
        for i in range(len(ejes)):
            ax[i].scatter(mh_all,ejes[i],color='k',alpha=0.7,s=5)
            ax[i].scatter(mh_all[no_sel],ejes_accu[i],color='red',alpha=0.7,s=5)
            ax[i].scatter(mh_all[no_m],ejes_m[i],color='green',alpha=0.7,s=25)
            ax[i].axhline(accu, color='r', linestyle='dashed', linewidth=3)
            ax[i].set_xlabel('$[H]$',fontsize=20)
            ax[i].set_ylabel(r'$\sigma_{\vec {v%s}}(mas)$'%(names[i]),fontsize=20)
            if i ==0:
                np.savetxt(pruebas+'dvx_mag_IN.txt',(mh_all,ejes[i]),fmt='%.4f',header='mh_all,dvx_all')
                np.savetxt(pruebas+'NO_dvx_mag_IN.txt',(mh_all[no_m],ejes_m[i]),fmt='%.4f',header='mh_all[no_m],dvx_all[no]')
    
    # In[7]:
    
    
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
        mu1 =2*umu1-1 # scale and shift to [-3., 3.)
        sigma1 = (usigma1)*3
        amp1 = uamp1    
    
        
        # mu2 = -0.018+ (0.062*umu2-0.031)# scale and shift to [-3., 3.)
        # sigma2 = 2.9+(0.15*usigma2-0.075)
        mu2= 2*umu2-1
        sigma2=5*usigma2
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
    #%%                              fig=plt.subplots(6, 6, figsize=(28, 28)))
    # This is de corner plot
    # fig, axes = dyplot.cornerplot(res, color='blue', show_titles=True, 
    #                               title_kwargs={'x': 0.65, 'y': 1.05}, labels=labels,
    #                               fig=plt.subplots(6, 6, figsize=(28, 28)))
    
    
    # plt.show() 
    
    
    # In[12]:
    
    
    res.summary()
    
    
    # In[13]:
    
    
    from dynesty import utils as dyfunc
    
    samples, weights = res.samples, np.exp(res.logwt - res.logz[-1])
    mean, cov = dyfunc.mean_and_cov(samples, weights)
    print(mean)
    
    
    # In[14]:
    
    
    plt.figure(figsize =(10,10))
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
    
    a=1#to chnge the axix a=-1
    h=plt.hist(v_y*a, bins= nbins, color='darkblue', alpha = 0.6, density =True, histtype = 'stepfilled')
    xplot = np.linspace(min(x), max(x), 100)
    
    # plt.plot(xplot, gaussian(xplot, mean[0], mean[1], mean[2]) , color="darkorange", linewidth=3, alpha=0.6)
    
    plt.plot(xplot, gaussian(xplot*a, mean[0], mean[1], mean[2]) + gaussian(xplot*a, mean[3], mean[4], mean[5]), color="darkorange", linewidth=3, alpha=0.6)
    plt.plot(xplot, gaussian(xplot*a, mean[0], mean[1], mean[2])  , color="k", linestyle='dashed', linewidth=3, alpha=0.6)
    plt.plot(xplot, gaussian(xplot*a, mean[3], mean[4], mean[5])  , color="red", linestyle='dashed', linewidth=3, alpha=0.6)
    plt.xlim(-15,15)
    # plt.axvline(mean[0],linestyle='dashed',color='orange')
    # plt.axvline(mean[3],linestyle='dashed',color='orange')
    plt.text(min(x),max(h[0]),'$\mu_{1}=%.3f$'%(mean[0]),color='k')
    plt.text(min(x),max(h[0]-0.01),'$\sigma_{1}=%.3f$'%(mean[1]),color='k')
    plt.text(min(x),max(h[0]-0.02),'$amp_{1}=%.3f$'%(mean[2]),color='k')
    plt.text(max(x)/2,max(h[0]),'$\mu_{2}=%.3f$'%(mean[3]),color='red')
    plt.text(min(x),max(h[0]-0.05),'$logz=%.0f$'%(results['logz'][-1]),color='b')
    # # if accu <10:
    # #     plt.text(min(x),max(h[0]-0.05),'$\sigma_{vy}<%.1f\ mas\ a^{-1}$'%(accu),color='b')
    plt.text(min(x),max(h[0]-0.04),'$nbins=%s$'%(nbins),color='b')
    plt.text(max(x)/2,max(h[0]-0.01),'$\sigma_{2}=%.3f$'%(mean[4]),color='red')
    plt.text(max(x)/2,max(h[0]-0.02),'$amp_{2}=%.3f$'%(mean[5]),color='red')
    # if (chip==2 or chip==3) and in_brick==1:
    #     plt.text(max(x)/2,max(h[0]-0.05),'$list = %.0f$'%(lst),color='b')
    # # elif in_brick==0:
    # #     if (chip==2 or chip==3):
    # #         plt.text(max(x)/2,max(h[0]-0.05),'$list =%.0f %s$'%(lst,'out'),color='b')
    # #     elif chip=='both':
    # #         plt.text(max(x)/2,max(h[0]-0.05),'$list =%s %s$'%(lst,'out'),color='b')
    plt.ylabel('N')
    # # plt.xlabel(r'$\mu_{l}$ (Km s$^{-1}$)') 
    plt.xlabel(r'$\mathrm{v_{b} (mas\ a^{-1})}$') 
    
    # # #%%
    pruebas='/Users/amartinez/Desktop/PhD/HAWK/The_Brick/photometry/pruebas/'
    
    if sloop==1:
        with open (pruebas+'brick_vy_gauss_var.txt', 'w') as f:
            f.write('%.4f %.4f %.4f %.4f %.4f %.4f %.0f %s'%(mean[0], mean[1], mean[2],mean[3], mean[4], mean[5],results['logz'][-1],nbins)+'\n')
    else:
        with open (pruebas+'brick_vy_gauss_var.txt', 'a') as f:
            f.write('%.4f %.4f %.4f %.4f %.4f %.4f %.0f %s'%(mean[0], mean[1], mean[2],mean[3], mean[4], mean[5],results['logz'][-1],nbins)+'\n')
 
#%%
pruebas='/Users/amartinez/Desktop/PhD/HAWK/The_Brick/photometry/pruebas/'

media=np.loadtxt(pruebas+'brick_vy_gauss_var.txt')#,delimiter=',')
va=['mu1','sigma1','amp1','mu2','sigma2','amp2']
for i in range(len(va)):
    print('%s = %.4f '%(va[i],np.average(media[:,i])))
    print('-'*20)
for i in range(len(va)):
    print('+'*20)
    print('d%s = %.4f'%(va[i],np.std(media[:,i])))
#%%
for i in range(len(va)):
    print('+'*20)
    print('sig_clip_d%s = %s'%(va[i],sigma_clipped_stats(media[:,i],sigma=1))) 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    