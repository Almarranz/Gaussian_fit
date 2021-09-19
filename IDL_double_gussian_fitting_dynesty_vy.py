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

band='H'
folder='im_jitter_NOgains/'
exptime=10
data='/Users/amartinez/Desktop/PhD/python/Gaussian_fit/'
tmp='/Users/amartinez/Desktop/PhD/HAWK/The_Brick/photometry/058_'+band+'/dit_'+str(exptime)+'/'+folder+'tmp_bs/'






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
rcParams.update({'font.size': 30})
rcParams.update({'figure.figsize':(10,5)})
rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})




chip='both'#can be 1 or 4 (refers to the chip on GNS fields)
field=20#fields can be 3 or 20 (refers to GNS fields)
nbins=21
select=1#uses the lists selec form alignment_with_GNS.pro script
if select ==0:
    gaussian='/Users/amartinez/Desktop/PhD/HAWK/The_Brick/photometry/058_'+band+'/dit_'+str(exptime)+'/'+folder+'Gaussian_fit/'
else:
    gaussian='/Users/amartinez/Desktop/PhD/HAWK/The_Brick/photometry/058_'+band+'/dit_'+str(exptime)+'/'+folder+'Gaussian_fit/select_'

accu=1.5
flds=[3,20,16]
chips=[1,4,2,3]
# flds=[16]
# chips=[3]
v_x=[]
v_y=[]
dvx=[]
dvy=[]
mh=[]
if chip =='both':
    for i in range(len(flds)):
        for j in range(len(chips)):
            try:
                print((flds[i],chips[j]))
                v_x0,v_y0,dvx0,dvy0,mh0=np.loadtxt(gaussian+'NPL058_IDL_mas_vx_vy_field%s_chip%s.txt'%(flds[i],chips[j]),unpack=True)
                v_x=np.r_[v_x,v_x0]
                v_y=np.r_[v_y,v_y0]
                dvx=np.r_[dvx,dvx0]
                dvy=np.r_[dvy,dvy0]
                mh=np.r_[mh,mh0]
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
    v_x,v_y,dvx,dvy,mh=np.loadtxt(gaussian+'NPL058_IDL_mas_vx_vy_field%s_chip%s.txt'%(field,chip),unpack=True)



select=np.where((dvy<accu) & (dvx<accu) )
v_y=v_y[select]
v_x=v_x[select]
mh_all=mh
mh=mh[select]
fig,ax=plt.subplots(1,1)
# sig_h=sigma_clip(v_y,sigma=1000,maxiters=20,cenfunc='mean',masked=True)
# v_y=v_y[sig_h.mask==False]
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


# In[6]:

ejes=[dvx,dvy]
names=['x','y']
if accu<5:
    fig, ax=plt.subplots(1,2,figsize=(20,10))
    for i in range(len(ejes)):
        ax[i].scatter(mh_all,ejes[i],color='k',alpha=0.7,s=5)
        ax[i].axhline(accu, color='r', linestyle='dashed', linewidth=3)
        ax[i].set_xlabel('$[H]$',fontsize=20)
        ax[i].set_ylabel(r'$\sigma_{\vec {v%s}}(mas)$'%(names[i]),fontsize=20)


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
    mu1 = 2* umu1-1  # scale and shift to [-10., 10.)
    sigma1 = 1.8* (usigma1+1)   
    amp1 = 1 * uamp1 

    
    mu2 = 0.4 * umu2-0.2
    sigma2 = 2 * usigma2   
    amp2 = 0.66* uamp2   
    

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
rcParams.update({'font.size': 25})


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


plt.figure(figsize =(8,8))
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
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})



results = sampler.results
print(results['logz'][-1])


h=plt.hist(v_y, bins= nbins, color='darkblue', alpha = 0.6, density =True, histtype = 'stepfilled')
xplot = np.linspace(min(x), max(x), 100)

# plt.plot(xplot, gaussian(xplot, mean[0], mean[1], mean[2]) , color="darkorange", linewidth=3, alpha=0.6)

plt.plot(xplot, gaussian(xplot, mean[0], mean[1], mean[2]) + gaussian(xplot, mean[3], mean[4], mean[5]), color="darkorange", linewidth=3, alpha=0.6)
plt.plot(xplot, gaussian(xplot, mean[0], mean[1], mean[2])  , color="red", linestyle='dashed', linewidth=3, alpha=0.6)
plt.plot(xplot, gaussian(xplot, mean[3], mean[4], mean[5])  , color="k", linestyle='dashed', linewidth=3, alpha=0.6)

# plt.axvline(mean[0],linestyle='dashed',color='orange')
# plt.axvline(mean[3],linestyle='dashed',color='orange')
plt.text(min(x),max(h[0]),'$\mu_{1}=%.3f$'%(mean[0]),color='red')
plt.text(min(x),max(h[0]-0.01),'$\sigma_{1}=%.3f$'%(mean[1]),color='red')
plt.text(min(x),max(h[0]-0.02),'$amp_{1}=%.3f$'%(mean[2]),color='red')
plt.text(max(x)/2,max(h[0]),'$\mu_{2}=%.3f$'%(mean[3]))
plt.text(min(x),max(h[0]-0.04),'$logz=%.0f$'%(results['logz'][-1]),color='b')
if accu <10:
    plt.text(min(x),max(h[0]-0.05),'$\sigma_{vy}<%.1f\ mas\ a^{-1}$'%(accu),color='b')
plt.text(max(x)/2,max(h[0]-0.04),'$nbins=%s$'%(nbins),color='b')
plt.text(max(x)/2,max(h[0]-0.01),'$\sigma_{2}=%.3f$'%(mean[4]))
plt.text(max(x)/2,max(h[0]-0.02),'$amp_{2}=%.3f$'%(mean[5]))

if chip=='both':
    plt.text(max(x)/2,max(h[0]-0.05),'$field%s,\ c%s$'%('All',chip),color='b')
else:
    plt.text(max(x)/2,max(h[0]-0.05),'$field%s,\ c%s$'%(field,chip),color='b')
# elif in_brick==0:
#     if (chip==2 or chip==3):
#         plt.text(max(x)/2,max(h[0]-0.05),'$field%s c%s$'%(field,chip),color='b')
#     elif chip=='both':
#         plt.text(max(x)/2,max(h[0]-0.05),'$field%s c%s$'%(field,chip),color='b')
plt.ylabel('$N$')
# plt.xlabel(r'$\mu_{l}$ (Km s$^{-1}$)') 
plt.xlabel('$v_{y} (mas\ yr^{-1}), IDL,\ Chip \ %s$'%(chip)) 

# #%%
# # Example data
# t = np.arange(0.0, 1.0 + 0.01, 0.01)
# s = np.cos(4 * np.pi * t) + 2

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.plot(t, s)

# plt.xlabel(r'\textbf{time} (s)')
# plt.ylabel(r'\textit{voltage} (mV)',fontsize=16)
# plt.title(r"\TeX\ is Number "
#           r"$\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!",
#           fontsize=16, color='gray')
# # Make room for the ridiculously large title.
# plt.subplots_adjust(top=0.8)

# plt.savefig('tex_demo')
# plt.show()
