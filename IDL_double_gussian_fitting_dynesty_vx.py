#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 13:41:58 2021

@author: amartinez
"""
#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import dynesty

from astropy.stats import sigma_clip
from astropy.stats import sigma_clipped_stats
# In[4]:

band='H'
folder='im_jitter_NOgains/'
exptime=10
data='/Users/amartinez/Desktop/PhD/python/Gaussian_fit/'
tmp='/Users/amartinez/Desktop/PhD/HAWK/The_Brick/photometry/058_'+band+'/dit_'+str(exptime)+'/'+folder+'tmp_bs/'

# In[4]:


# plt.rcParams['figure.figsize'] = (20,10)
#from matplotlib import rc
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
rcParams.update({
    "text.usetex": False,
    "font.family": "sans",
    "font.sans-serif": ["Palatino"]})

from matplotlib import rc
# In[5]:
auto='auto'
if auto =='auto':
    step=np.arange(1.5,2.5,0.5)#
else:
    step=np.arange(0.5,0.7,0.1)#also works if running each bing width one by one, for some reason...
print(step)
media_amp=[]
zone='Z1'

#%%
# for sloop in range(ran,ran+1):
for sloop in range(len(step)):
    
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    show_field='no'
    chip=3 #can be 1 or 4 (refers to the chip on GNS fields)
    field=16 #fields can be 3 or 20 (refers to GNS fields)
    sm=0.5
    
    gaussian='/Users/amartinez/Desktop/PhD/HAWK/The_Brick/photometry/058_'+band+'/dit_'+str(exptime)+'/'+folder+'Gaussian_fit/'


    # nbins=25+sloop
    
    accu=2
    
    # flds=[16,3,7]#I feel that field 10 make things worse for some reason
    flds=[16]#I feel that field 10 make things worse for some reason

    # chips=[1,2,3,4]
    # flds=[3]
    chips=[3]
    v_x=[]
    v_y=[]
    dvx=[]
    dvy=[]
    mh=[]
    m=[]
    af=[]
    bc=[]
    if chip =='both':
        for i in range(len(flds)):
            for j in range(len(chips)):
                try:
                    
                    v_x0,v_y0,dvx0,dvy0,mh0,m0=np.loadtxt(gaussian+'aa_NPL058_IDL_mas_vx_vy_field%s_chip%s.txt'%(flds[i],chips[j]),unpack=True)
                    v_x=np.r_[v_x,v_x0]
                    v_y=np.r_[v_y,v_y0]
                    dvx=np.r_[dvx,dvx0]
                    dvy=np.r_[dvy,dvy0]
                    mh=np.r_[mh,mh0]
                    m=np.r_[m,m0]
                    print((flds[i],chips[j]))
                    af.append([flds[i]])
                    bc.append([chips[j]])
                except:
                    print('NO hay lista field%s, chip%s'%(flds[i],chips[j]))
                    pass
      
    # if chip =='both':
    #     v_x1,v_y1,dvx1,dvy1,mh1=np.loadtxt(gaussian+'NPL058_IDL_mas_vx_vy_field20_chip1.txt',unpack=True)
    #     v_x2,v_y2,dvx2,dvy2,mh2=np.loadtxt(gaussian+'NPL058_IDL_mas_vx_vy_field20_chip4.txt',unpack=True)
    #     v_x3,v_y3,dvx3,dvy3,mh3=np.loadtxt(gaussian+'NPL058_IDL_mas_vx_vy_field3_chip1.txt',unpack=True)
    #     v_x4,v_y4,dvx4,dvy4,mh4=np.loadtxt(gaussian+'NPL058_IDL_mas_vx_vy_field3_chip4.txt',unpack=True)
        
    #     v_x=np.r_[v_x1,v_x2,v_x3,v_x4]
    #     v_y=np.r_[v_y1,v_y2,v_y3,v_y4]
    #     dvx=np.r_[dvx1,dvx2,dvx3,dvx4]
    #     dvy=np.r_[dvy1,dvy2,dvy3,dvy4]
    #     mh=np.r_[mh1,mh2,mh3,mh4]
    else :
        v_x,v_y,dvx,dvy,mh,m,ar,dec,arg,decg=np.loadtxt(gaussian+'%s_aa_NPL058_IDL_mas_vx_vy_field%s_chip%s.txt'%(zone,field,chip),unpack=True)
    mh_all=mh
    m_all=m
    dvx_all=dvx
    dvy_all=dvy
    
    max_M=22.5
    sel_M=np.where(abs(mh)<max_M)
    v_x=v_x[sel_M]
    v_y=v_y[sel_M]
    mh=mh[sel_M]
    m=m[sel_M]
    dvx=dvx[sel_M]
    dvy=dvy[sel_M]
    
    
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
#%%

    fig,ax=plt.subplots(1,1)
    sig_h=sigma_clip(v_x,sigma=60,maxiters=20,cenfunc='mean',masked=True)
    v_x=v_x[sig_h.mask==False]
    # h=ax.hist(v_x,bins=nbins,edgecolor='black',linewidth=2,density=True)
    # h=ax.hist(v_x,bins=list_bin,edgecolor='black',linewidth=2,density=True)
    if auto != 'auto':
        list_bin=np.arange(min(v_x),max(v_x),step[sloop])
        auto=list_bin
        print(list_bin)
        nbins=len(list_bin)-1
        print(30*'#'+'\n'+'nbins=%s'%(nbins)+'\n'+30*'#')
    h=ax.hist(v_x,bins=auto,edgecolor='black',linewidth=2,density=True)
    h1=np.histogram(v_x,bins=auto,density=False)
    print(35*'-'+'\n'+'The width of the bin is: %.3f'%(h[1][3]-h[1][2])+'\n'+35*'-')

    x=[h[1][i]+(h[1][1]-h[1][0])/2 for i in range(len(h[0]))]#middle value for each bin
    ax.axvline(np.mean(v_x), color='r', linestyle='dashed', linewidth=3)
    ax.legend(['Chip=%s, %s, mean= %.2f, std=%.2f'
                  %(chip,len(v_x),np.mean(v_x),np.std(v_x))],fontsize=12,markerscale=0,shadow=True,loc=1,handlelength=-0.0)
    y=h[0]#height for each bin
    # for yi in range(len(y)):
    #     if y[yi]==0:
    #         y[yi]+=0.00001
    #     yerr = np.sqrt(h1[0][yi])/(len(v_y)*((h1[1][3]-h1[1][2])))
    yerr=0.0001
    y += yerr
    ax.scatter(x,y,color='g',zorder=3)
    
    
    # In[6]:
    count=0
    for i in range(len(mh)):
        if abs(mh_all[i]-m_all[i])>sm:
            count+=1
    print(35*'#'+'\n'+'stars with diff in mag > %s: %s'%(sm,count)+'\n'+35*'#')
    # In[ ]:
    
    ejes=[dvx_all,dvy_all]
    no_sel=np.where((dvx_all>accu)&(dvy_all>accu))
    no_m=np.where(abs(mh_all-m_all)>sm)
    ejes_accu=[dvx_all[no_sel],dvy_all[no_sel]]
    ejes_m=[dvx_all[no_m],dvy_all[no_m]]
    names=['x','y']
    if accu<5:
        fig, ax=plt.subplots(1,2,figsize=(20,10))
        for i in range(len(ejes)):
            ax[i].scatter(mh_all,ejes[i],color='k',alpha=0.7,s=5)
            ax[i].scatter(mh_all[no_sel],ejes_accu[i],color='red',alpha=0.7,s=5)
            ax[i].scatter(mh_all[no_m],ejes_m[i],color='green',alpha=0.7,s=25)
            ax[i].axhline(accu, color='r', linestyle='dashed', linewidth=3)
            ax[i].axvline(max_M, color='r', linestyle='dashed', linewidth=3)
            ax[i].set_xlabel('$[H]$',fontsize=20)
            ax[i].set_ylabel(r'$\sigma_{\vec {v%s}}(mas)$'%(names[i]),fontsize=20)
    
    
    
    # In[7]:
    
    
    def gaussian(x, mu, sig, amp):
        return amp * (1 / (sig * (np.sqrt(2 * np.pi)))) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    
    
    # In[8]:
    
    
    # In[5]:
    
    
    def loglike(theta):
        mu1, sigma1, amp1,mu2,sigma2,amp2 = theta
        model = gaussian(x, mu1, sigma1, amp1)+gaussian(x,mu2,sigma2,amp2)
     
        return -0.5 * np.sum(((y - model)/yerr) ** 2)#chi squared model
    
    
    # In[6]:
    
    
    def prior_transform(utheta):
        """Transforms the uniform random variable `u ~ Unif[0., 1.)`
        to the parameter of interest `x ~ Unif[-10., 10.)`."""
        #x = 2. * u - 1.  # scale and shift to [-1., 1.)
        #x *= 10.  # scale to [-10., 10.)
        umu1, usigma1, uamp1,  umu2, usigma2, uamp2= utheta
    
    #     mu1 = -1. * umu1-8   # scale and shift to [-10., 10.)
        mu1 =-5*umu1  # scale and shift to [-3., 3.)
        sigma1 = 5*(usigma1)
        amp1 = uamp1
        
        # mu2 = 0.12+((umu2*0.06)-0.03)
        mu2=4*umu2-2
        # sigma2 =3.3+((usigma2*0.33)-0.33/2)
        sigma2=5*usigma2
        amp2 = uamp2*1                                                   
        # amp2=0.54 +(uamp2*0.06-0.06/2)
        
       
        
    
        return mu1, sigma1, amp1, mu2, sigma2, amp2
    
    
    # In[8]:
    
    
    sampler = dynesty.NestedSampler(loglike, prior_transform, ndim=6, nlive=200,
                                            bound='multi', sample='rwalk')
    sampler.run_nested()
    res = sampler.results
    
    
    # In[13]:
    
    
    
    
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
    rcParams.update({'font.size': 15})
    
    
    # truths = [mu1_true, sigma1_true, amp1_true, mu2_true, sigma2_true, amp2_true]
    labels = [r'$mu1$', r'$sigma1$', r'$amp1$', r'$mu2$', r'$sigma2$', r'$amp2$',r'$mu3$', r'$sigma3$', r'$amp3$']
    # fig, axes = dyplot.traceplot(sampler.results, truths=truths, labels=labels,
    #                              fig=plt.subplots(6, 2, figsize=(16, 27)))
    
    fig, axes = dyplot.traceplot(sampler.results,labels=labels,
                                 fig=plt.subplots(6, 2, figsize=(16, 16)))
    plt.show()
    
    
    # In[11]:
    
    
    # fig, axes = dyplot.cornerplot(res, truths=truths, color='blue', show_titles=True, 
    #                               title_kwargs={'x': 0.65, 'y': 1.05}, labels=labels,
    #                               fig=plt.subplots(6, 6, figsize=(28, 28)))
    #Thsi is the corner plot
    
    # fig, axes = dyplot.cornerplot(res, color='blue', show_titles=True, 
    #                               title_kwargs={'x': 0.65, 'y': 1.05}, labels=labels,
    #                               fig=plt.subplots(9, 9, figsize=(28, 28)))
    
    
    plt.show() 
    
    
    # In[12]:
    
    
    res.summary()
    
    
    # In[13]:
    
    
    from dynesty import utils as dyfunc
    
    samples, weights = res.samples, np.exp(res.logwt - res.logz[-1])
    mean, cov = dyfunc.mean_and_cov(samples, weights)
    print(mean)
    rcParams.update({'font.size': 20})
    
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
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = 'dejavuserif'
    plt.rcParams['text.usetex'] = False
    from matplotlib import rc
    rc('font',**{'family':'serif','serif':['Palatino']})
    
    results = sampler.results
    print(results['logz'][-1])
    
    # h=plt.hist(v_x*-1, bins= nbins, color='darkblue', alpha = 0.6, density =True, histtype = 'stepfilled')
    h=plt.hist(v_x*-1, bins= auto, color='darkblue', alpha = 0.6, density =True, histtype = 'stepfilled')

    xplot = np.linspace(min(x), max(x), 100)
    
    # plt.plot(xplot, gaussian(xplot, mean[0], mean[1], mean[2]) , color="darkorange", linewidth=3, alpha=0.6)
    
    plt.plot(xplot, gaussian(xplot*-1, mean[0], mean[1], mean[2]) + gaussian(xplot*-1, mean[3], mean[4], mean[5])
             , color="darkorange", linewidth=3, alpha=1)
    plt.plot(xplot, gaussian(xplot*-1, mean[0], mean[1], mean[2])  , color="yellow", linestyle='dashed', linewidth=3, alpha=0.6)
    plt.plot(xplot, gaussian(xplot*-1, mean[3], mean[4], mean[5])  , color="red", linestyle='dashed', linewidth=3, alpha=0.6)
    plt.xlim(-15,15)
    plt.gca().invert_xaxis()
    # plt.axvline(mean[0],linestyle='dashed',color='orange')
    # plt.axvline(mean[3],linestyle='dashed',color='orange')
    # plt.text(min(x),max(h[0]),'$\mu_{1}=%.3f$'%(mean[0]),color='green')
    # plt.text(min(x),max(h[0]-0.01),'$\sigma_{1}=%.3f$'%(mean[1]),color='green')
    # plt.text(min(x),max(h[0]-0.02),'$amp_{1}=%.3f$'%(mean[2]),color='green')
    # plt.text(max(x)/2,max(h[0]),'$\mu_{2}=%.3f$'%(mean[3]),color='red')
    # plt.text(max(x)/2,max(h[0]-0.01),'$\sigma_{2}=%.3f$'%(mean[4]),color='red')
    # plt.text(max(x)/2,max(h[0]-0.02),'$amp_{2}=%.3f$'%(mean[5]),color='red')
    # plt.text(max(x)/2,max(h[0]-0.03),'$\mu_{3}=%.3f$'%(mean[6]))
    # plt.text(max(x)/2,max(h[0]-0.04),'$\sigma_{3}=%.3f$'%(mean[7]))
    # plt.text(max(x)/2,max(h[0]-0.045),'$amp_{3}=%.3f$'%(mean[8]))
    
    plt.text(max(x)/2,max(h[0]/2)-0.01,'$logz=%.0f$'%(results['logz'][-1]),color='b')
    # # if accu<10:
    # #     plt.text(min(x),max(h[0]/2)-0.005,'$\sigma_{vx}<%.1f\ mas\ a^{-1}$'%(accu),color='b')
    plt.text(max(x)/2,max(h[0]/2)-0.020,'$nbins=%s$'%(len(h[0])),color='b')
    # plt.text(min(x),max(h[0]/2)-0.030,'$diff\ mag < %s$'%(sm),color='b')
    # if show_field=='yes':
    #     if chip=='both':
    #         plt.text(max(x)/2,max(h[0]-0.06),'$field%s$'%(af),color='b')
    #         plt.text(max(x)/2,max(h[0]-0.07),'$chip%s$'%(bc),color='b')
    #     else:
    #         plt.text(max(x)/2,max(h[0]-0.06),'$field%s,\ c%s$'%(field,chip),color='b')
    plt.ylabel('N')
    plt.legend(['Zone B'],fontsize=20,markerscale=0,shadow=True,loc=2,handlelength=-0.0)
    # plt.xlabel(r'$\mu_{l}$ (Km s$^{-1}$)') 
    plt.xlabel(r'$\mathrm{\mu_{l} (mas\ a^{-1})}$') 

    pruebas='/Users/amartinez/Desktop/PhD/HAWK/The_Brick/photometry/pruebas/'
#%%
#for file in range(1,4):
    if sloop==0:
        with open (pruebas+'%s_vx_gauss_var.txt'%(zone), 'w') as f:
            f.write('%.4f %.4f %.4f %.4f %.4f %.4f %.0f %s'%(mean[0], mean[1], mean[2],mean[3], mean[4], mean[5],results['logz'][-1],nbins)+'\n')
    else:
        with open (pruebas+'%s_vx_gauss_var.txt'%(zone), 'a') as f:
            f.write('%.4f %.4f %.4f %.4f %.4f %.4f %.0f %s'%(mean[0], mean[1], mean[2],mean[3], mean[4], mean[5],results['logz'][-1],nbins)+'\n')


#%%
pruebas='/Users/amartinez/Desktop/PhD/HAWK/The_Brick/photometry/pruebas/'

media=np.loadtxt(pruebas+'%s_vx_gauss_var.txt'%(zone))#,delimiter=',')
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

# In[ ]:

print(sampler.citations)


