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
from matplotlib.ticker import FormatStrFormatter
import scipy.integrate as integrate

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
# step=np.arange(1.25,2,0.25)
# print(step)
# list_bin=np.arange(-15,15+step[0],step[0])
# print(list_bin)
# media_amp=[]
auto='auto'
if auto =='auto':
    step=np.arange(0,1,1)#also works if running each bing width one by one, for some reason...
else:
    step=np.arange(0.79,1.0901,0.1)#also works if running each bing width one by one, for some reason...
#%%

# In[5]:
 # this read a txt file indicating the list of common stars used for the alignments
#There are 3 different lists. 1,2 and 3. Being # 3 the smaller and more 'accured' within the brick
#This is generated en 13_alig....
#chip='both'
ran=0
for sloop in range(len(step)):
    sm=0.45
    chip='both'
    
    in_brick=1#slect stars on the brick, if =1 or out of brick if =1.
    
    # nbins=9
    
    accu=2# select stars cutting by uncertainty. With a large value all star are selected
    if in_brick==1:
        if chip =='both':
            # v_x2,v_y2,dvx2,dvy2,mh2,m2=np.loadtxt(data+'IDL_arcsec_vx_vy_chip2.txt',unpack=True)
            # v_x3,v_y3,dvx3,dvy3,mh3,m3=np.loadtxt(data+'IDL_arcsec_vx_vy_chip3.txt',unpack=True)
            v_x2,v_y2,dvx2,dvy2,mh2,m2,ar,dec,arg,decg=np.loadtxt(data+'UP_aa_IDL_arcsec_vx_vy_chip3.txt',unpack=True)
            v_x3,v_y3,dvx3,dvy3,mh3,m3,ar,dec,arg,decg=np.loadtxt(data+'DOWN_aa_IDL_arcsec_vx_vy_chip2.txt',unpack=True)

            v_x=np.r_[v_x2,v_x3]
            v_y=np.r_[v_y2,v_y3]
            dvx=np.r_[dvx2,dvx3]
            dvy=np.r_[dvy2,dvy3]
            mh=np.r_[mh2,mh3]
            m=np.r_[m2,m3]
        elif chip==2 or chip==3:
            # lst=np.loadtxt(tmp+'IDL_lst_chip%s.txt'%(chip))
            lst=np.loadtxt(tmp+'aa_IDL_lst_chip%s.txt'%(chip))
            # v_x,v_y,dvx,dvy,mh,m=np.loadtxt(data+'IDL_arcsec_vx_vy_chip%s.txt'%(chip),unpack=True)
            v_x,v_y,dvx,dvy,mh,m,ar,dec,arg,decg=np.loadtxt(data+'aa_IDL_arcsec_vx_vy_chip%s.txt'%(chip),unpack=True)
    elif in_brick==0:
         if chip=='both':
            lst='All '
            v_x10,v_y10,dvx10,dvy10,mh10=np.loadtxt(data+'IDL_arcsec_vx_vy_chip2_out_Brick10.txt',unpack=True)
            #v_x12,v_y12,dvx12,dvy12,mh12=np.loadtxt(data+'IDL_arcsec_vx_vy_chip3_out_Brick12.txt',unpack=True)
            v_x16,v_y16,dvx16,dvy16,mh16=np.loadtxt(data+'IDL_arcsec_vx_vy_chip3_out_Brick16.txt',unpack=True)
            
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
         else:
            lst=3
            v_x,v_y,dvx,dvy,mh=np.loadtxt(data+'IDL_arcsec_vx_vy_chip%s_out_Brick16.txt'%(chip),unpack=True)
    # select=np.where((dvx<accu)&(dvy<accu))
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
    sig_h=sigma_clip(v_x,sigma=500,maxiters=20,cenfunc='mean',masked=True)#an outlier on vy is not(genarally) also on vx
    v_x=v_x[sig_h.mask==False]
    if auto != 'auto':
        list_bin=np.arange(min(v_y),max(v_y),step[sloop])
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
    # These are two different way of computing the error for the normalized bins. None of then seem to work very well
# =============================================================================
#     y1=h1[0]
#     yerr=[]
#     y1=np.where(y1==0,0.001,y1)
#     yerr = [np.sqrt(y1[yi])/(len(v_x)*100*((h1[1][3]-h1[1][2]))) for yi in range(len(y))]
# =============================================================================
    yerr=[]
    y=np.where(y==0,0.001,y)
    y1=h1[0]
    y1=np.where(y1==0,0.001,y1)
    # yerr = y*np.sqrt(1/y1+1/len(v_x))
    yerr = y*np.sqrt(1/y1)
# =============================================================================
#     yerr=[]
#     y=np.where(y==0,0.0001,y)
#     y1=h1[0]
#     y1=np.where(y1==0,0.001,y1)
#     # yerr = y*np.sqrt(1/y1+1/len(v_x))
#     yerr=0.0001
#     y += yerr
#     # ax.scatter(x,y,color='g',zorder=3)
# =============================================================================
   
    
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
    #%%
    count=0
    for i in range(len(mh)):
        if abs(mh_all[i]-m_all[i])>sm:
            count+=1
    print(35*'#'+'\n'+'stars with diff in mag > %s: %s'%(sm,count)+'\n'+35*'#')
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
        mu1 = -4*umu1   # scale and shift to [-3., 3.)
        sigma1 = 3*(usigma1)
        amp1 = uamp1*1
        
        
        #mu2 = -0.16 + (0.16*umu2-0.08)  
        mu2 = 6*umu2-3  

        # sigma2 = 3.60 +  (0.5*usigma2-0.5/2)
        sigma2=usigma2*4
        # amp2 = 0.42 + (0.08*uamp2-0.04)
        # amp2 = 0.59 + (0.08*uamp2-0.04)
        amp2 = uamp2*1
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
    #                               fig=plt.subplots(6, 6, figsize=(28, 28)))
    
    fig, axes = dyplot.cornerplot(res, color='royalblue', show_titles=True, 
                                  title_kwargs={'x': 0.65, 'y': 1.05}, labels=labels,
                                  fig=plt.subplots(6, 6, figsize=(28, 28)))
    plt.legend(['Zone A, $\mu_{l}$'],fontsize=70,markerscale=0,shadow=True,bbox_to_anchor=(1,6.5),handlelength=-0.0)

    
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
    from matplotlib import rcParams
    # 
    results = sampler.results
    print(results['logz'][-1])
    
    fig, ax = plt.subplots(figsize=(8,8))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    h=plt.hist(v_x*-1, bins= auto, color='royalblue', alpha = 0.5, density =True, histtype = 'stepfilled')
    xplot = np.linspace(min(x), max(x), 100)
    
    # plt.plot(xplot, gaussian(xplot, mean[0], mean[1], mean[2]) , color="darkorange", linewidth=3, alpha=0.6)
    
    plt.plot(xplot, gaussian(xplot*-1, mean[0], mean[1], mean[2]) + gaussian(xplot*-1, mean[3], mean[4], mean[5]), color="darkorange", linewidth=3, alpha=1)
    plt.plot(xplot, gaussian(xplot*-1, mean[0], mean[1], mean[2])  , color="yellow", linestyle='dashed', linewidth=2, alpha=1)
    plt.plot(xplot, gaussian(xplot*-1, mean[3], mean[4], mean[5])  , color="red", linestyle='dashed', linewidth=2, alpha=1)
    
    # plt.axvline(mean[0],linestyle='dashed',color='orange')
    # plt.axvline(mean[3],linestyle='dashed',color='orange')
    # plt.text(min(x),max(h[0]),'$\mu_{1}=%.3f$'%(mean[0]),color='k')
    # plt.text(min(x),max(h[0]-0.01),'$\sigma_{1}=%.3f$'%(mean[1]),color='k')
    # plt.text(min(x),max(h[0]-0.02),'$amp_{1}=%.3f$'%(mean[2]),color='k')
    # plt.text(max(x)/2,max(h[0]),'$\mu_{2}=%.3f$'%(mean[3]),color='red')
    plt.text(min(x),max(h[0]-0.04),'$logz=%.0f$'%(results['logz'][-1]),color='b')
    # if accu<10:
    #     plt.text(min(x),max(h[0]-0.05),'$\sigma_{vx}<%.1f\ mas\ a^{-1}$'%(accu),color='b')
    plt.text(max(x)/2,max(h[0]-0.04),'$nbins=%s$'%(len(h[0])),color='b')
    # plt.text(max(x)/2,max(h[0]-0.01),'$\sigma_{2}=%.3f$'%(mean[4]),color='red')
    # plt.text(max(x)/2,max(h[0]-0.02),'$amp_{2}=%.3f$'%(mean[5]),color='red')
    # if (chip==2 or chip==3) and in_brick==1:
    #     plt.text(max(x)/2,max(h[0]-0.05),'$list = %.0f$'%(lst))
    # elif in_brick==0:
    #     plt.text(max(x)/2,max(h[0]-0.05),'$list = %s$'%('out Brick'))
    plt.legend(['Zone A'],fontsize=20,markerscale=0,shadow=True,loc=2,handlelength=-0.0)    
    plt.xlim(-15,15) 
    plt.gca().invert_xaxis()
    plt.ylabel('N')
    # plt.xlabel(r'$\mu_{l}$ (Km s$^{-1}$)') 
    plt.xlabel(r'$\mathrm{\mu_{l} (mas\ a^{-1})}$') 
    
    fun1= lambda x: (mean[2] * (1 / (mean[1] * (np.sqrt(2 * np.pi)))) * np.exp(-np.power(x - mean[0], 2.) / (2 * np.power(mean[1], 2.))) )
    # result = integrate.quad(gaussian(x, mean[0], mean[1], mean[2]),-15,15)
    gau1=integrate.quad(fun1,-15,15)
    
    fun2= lambda x: (mean[5] * (1 / (mean[4] * (np.sqrt(2 * np.pi)))) * np.exp(-np.power(x - mean[3], 2.) / (2 * np.power(mean[4], 2.))) )
    # result = integrate.quad(gaussian(x, mean[0], mean[1], mean[2]),-15,15)
    gau2=integrate.quad(fun2,-15,15)
    
    # fun3= lambda x: (mean[8] * (1 / (mean[7] * (np.sqrt(2 * np.pi)))) * np.exp(-np.power(x - mean[6], 2.) / (2 * np.power(mean[7], 2.))) )
    # # result = integrate.quad(gaussian(x, mean[0], mean[1], mean[2]),-15,15)
    # gau3=integrate.quad(fun3,-15,15)
    # media_amp.append(gau1[0])
    # print(gau1[0],gau2[0])
    # # print(30*'&'+'\n'+'Area under Gaus1:%s'%(gau1[0])+'\n'+'Area under Gaus2:%s'(gau2[0])+'\n'+30*'&',)
    # print(30*'&')
    # print('Area under Gaus1:%.3f'%(gau1[0]))
    # print('Area under Gaus2:%.3f'%(gau2[0]))
    # print('Total area = %.3f'%(gau1[0]+gau2[0]))
    # print(30*'&')
    
    if sloop==0:
        with open (pruebas+'brick_vx_2gauss_var.txt', 'w') as f:
            f.write('%.4f %.4f %.4f %.4f %.4f %.4f %.0f %s'%(mean[0], mean[1], mean[2],mean[3], mean[4], mean[5],results['logz'][-1],len(h[0]))+'\n')
    else:
        with open (pruebas+'brick_vx_2gauss_var.txt', 'a') as f:
            f.write('%.4f %.4f %.4f %.4f %.4f %.4f %.0f %s'%(mean[0], mean[1], mean[2],mean[3], mean[4], mean[5],results['logz'][-1],len(h[0]))+'\n')
    
#%%
# =============================================================================
# pruebas='/Users/amartinez/Desktop/PhD/HAWK/The_Brick/photometry/pruebas/'
# 
# media=np.loadtxt(pruebas+'brick_vx_gauss_var.txt')#,delimiter=',')
# va=['mu1','sigma1','amp1','mu2','sigma2','amp2']
# print('Media area broad = %.3f'%np.average(media_amp))
# for i in range(len(va)):
#     print('%s = %.4f '%(va[i],np.average(media[:,i])))
#     print('-'*20)
# for i in range(len(va)):
#     print('+'*20)
#     print('d%s = %.4f'%(va[i],np.std(media[:,i])))
# =============================================================================
#%%
# for i in range(len(va)):
#     print('+'*20)
#     print('sig_clip_d%s = %s'%(va[i],sigma_clipped_stats(media[:,i],sigma=1))) 
    
fig, ax = plt.subplots(figsize=(8,8))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# mean=[-1.91049268333333,	1.12003936,	0.18239401,	0.223946633333333,	3.47003635,	0.822778293333333,]
h=plt.hist(v_x*-1, bins= 21, color='royalblue', alpha = 0.6, density =True, histtype = 'stepfilled')

xplot = np.linspace(-15, 15, 100)

# plt.plot(xplot, gaussian(xplot, mean[0], mean[1], mean[2]) , color="darkorange", linewidth=3, alpha=0.6)

plt.plot(xplot, gaussian(xplot*-1, mean[0], mean[1], mean[2]) + gaussian(xplot*-1, mean[3], mean[4], mean[5]), color="darkorange", linewidth=3, alpha=1)

plt.plot(xplot, gaussian(xplot*-1, mean[0], mean[1], mean[2])  , color="yellow", linestyle='dashed', linewidth=2, alpha=1)
plt.plot(xplot, gaussian(xplot*-1, mean[3], mean[4], mean[5])  , color="red", linestyle='dashed', linewidth=2, alpha=1)
plt.xlim(-15,15)
# plt.ylim(-0,0.15)
plt.gca().invert_xaxis()
  
plt.ylabel('N')
plt.legend(['Brick Zone'],fontsize=20,markerscale=0,shadow=True,loc=2,handlelength=-0.0)
# plt.xlabel(r'$\mu_{l}$ (Km s$^{-1}$)') 
plt.xlabel(r'$\mathrm{\mu_{l} (mas\ a^{-1})}$')    

#%%


samples, weights = res.samples, np.exp(res.logwt - res.logz[-1])
mean, cov = dyfunc.mean_and_cov(samples, weights)
# print(mean)
quantiles = [dyfunc.quantile(samps, [0.16,0.5,0.84], weights=weights)
             for samps in samples.T]

# for i in range(6):
#     print(mean[i],quantiles[i])

#%%

for i in range(6):
    print('mean %.2f -+ %2f %2f'%(mean[i],quantiles[i][1]-quantiles[i][0],quantiles[i][2]-quantiles[i][1]))
    if i==0:
        with open (pruebas+'brick_vx_erros.txt', 'w') as f:
            f.write('%.2f %.2f %.2f'%(quantiles[i][1],quantiles[i][1]-quantiles[i][0],quantiles[i][2]-quantiles[i][1])+'\n')
    else:
        with open (pruebas+'brick_vx_erros.txt', 'a') as f:
           f.write('%.2f %.2f %.2f'%(quantiles[i][1],quantiles[i][1]-quantiles[i][0],quantiles[i][2]-quantiles[i][1])+'\n')

    