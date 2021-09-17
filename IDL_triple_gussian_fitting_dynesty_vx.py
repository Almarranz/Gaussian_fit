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
tmp='/Users/amartinez/Desktop/PhD/HAWK/The_Brick/photometry/054_'+band+'/dit_'+str(exptime)+'/'+folder+'tmp_bs/'

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
rcParams.update({'font.size': 30})
rcParams.update({'figure.figsize':(10,5)})
rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})


# In[5]:
# from matplotlib import rc
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

chip=1 #can be 1 or 4 (refers to the chip on GNS fields)
field=20 #fields can be 3 or 20 (refers to GNS fields)
gaussian='/Users/amartinez/Desktop/PhD/HAWK/The_Brick/photometry/058_'+band+'/dit_'+str(exptime)+'/'+folder+'Gaussian_fit/'

nbins=25
accu=1.1




if chip =='both':
    v_x1,v_y1,dvx1,dvy1,mh1=np.loadtxt(gaussian+'NPL058_IDL_mas_vx_vy_field20_chip1.txt',unpack=True)
    v_x2,v_y2,dvx2,dvy2,mh2=np.loadtxt(gaussian+'NPL058_IDL_mas_vx_vy_field20_chip4.txt',unpack=True)
    v_x3,v_y3,dvx3,dvy3,mh3=np.loadtxt(gaussian+'NPL058_IDL_mas_vx_vy_field3_chip1.txt',unpack=True)
    v_x4,v_y4,dvx4,dvy4,mh4=np.loadtxt(gaussian+'NPL058_IDL_mas_vx_vy_field3_chip4.txt',unpack=True)
    
    v_x=np.r_[v_x1,v_x2,v_x3,v_x4]
    v_y=np.r_[v_y1,v_y2,v_y3,v_y4]
    dvx=np.r_[dvx1,dvx2,dvx3,dvx4]
    dvy=np.r_[dvy1,dvy2,dvy3,dvy4]
    mh=np.r_[mh1,mh2,mh3,mh4]
else :
    v_x,v_y,dvx,dvy,mh=np.loadtxt(gaussian+'NPL058_IDL_mas_vx_vy_field%s_chip%s.txt'%(field,chip),unpack=True)

select=np.where((dvx<accu)&(dvy<accu))
v_x=v_x[select]
v_y=v_y[select]
mh_all=mh
mh=mh[select]
fig,ax=plt.subplots(1,1)
sig_h=sigma_clip(v_x,sigma=60,maxiters=20,cenfunc='mean',masked=True)
v_x=v_x[sig_h.mask==False]
h=ax.hist(v_x,bins=nbins,edgecolor='black',linewidth=2,density=True)
x=[h[1][i]+(h[1][1]-h[1][0])/2 for i in range(len(h[0]))]#middle value for each bin
ax.axvline(np.mean(v_x), color='r', linestyle='dashed', linewidth=3)
ax.legend(['Chip=%s, %s, mean= %.2f, std=%.2f'
              %(chip,len(v_x),np.mean(v_x),np.std(v_x))],fontsize=12,markerscale=0,shadow=True,loc=1,handlelength=-0.0)
y=h[0]#height for each bin
#yerr = y*0.05
#yerr = y*0.01
yerr=0.0001
y += yerr
ax.scatter(x,y,color='g',zorder=3)


# In[6]:


# In[ ]:

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


# In[5]:


def loglike(theta):
    mu1, sigma1, amp1,mu2,sigma2,amp2,mu3,sigma3,amp3 = theta
    model = gaussian(x, mu1, sigma1, amp1)+gaussian(x,mu2,sigma2,amp2)+gaussian(x,mu3,sigma3,amp3)
 
    return -0.5 * np.sum(((y - model)/yerr) ** 2)#chi squared model


# In[6]:


def prior_transform(utheta):
    """Transforms the uniform random variable `u ~ Unif[0., 1.)`
    to the parameter of interest `x ~ Unif[-10., 10.)`."""
    #x = 2. * u - 1.  # scale and shift to [-1., 1.)
    #x *= 10.  # scale to [-10., 10.)
    umu1, usigma1, uamp1,  umu2, usigma2, uamp2, umu3, usigma3, uamp3= utheta

#     mu1 = -1. * umu1-8   # scale and shift to [-10., 10.)
    mu1 =-3*umu1  # scale and shift to [-3., 3.)
    sigma1 = 3*(usigma1)
    amp1 = uamp1*1.5
    
    mu2 = 2*umu2-1
    sigma2 =1.79*(usigma2+1)
    #sigma2 =3.5*usigma2
    amp2 = uamp2*1
    
    mu3 =3*(umu3+1) # scale and shift to [-3., 3.)
    sigma3 = 2*(usigma3+0)
    amp3 = uamp3*1.5
    
    

    return mu1, sigma1, amp1, mu2, sigma2, amp2, mu3, sigma3, amp3


# In[8]:


sampler = dynesty.NestedSampler(loglike, prior_transform, ndim=9, nlive=200,
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
rcParams.update({'font.size': 25})


# truths = [mu1_true, sigma1_true, amp1_true, mu2_true, sigma2_true, amp2_true]
labels = [r'$mu1$', r'$sigma1$', r'$amp1$', r'$mu2$', r'$sigma2$', r'$amp2$',r'$mu3$', r'$sigma3$', r'$amp3$']
# fig, axes = dyplot.traceplot(sampler.results, truths=truths, labels=labels,
#                              fig=plt.subplots(6, 2, figsize=(16, 27)))

fig, axes = dyplot.traceplot(sampler.results,labels=labels,
                             fig=plt.subplots(9, 2, figsize=(16, 20)))
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

h=plt.hist(v_x, bins= nbins, color='darkblue', alpha = 0.6, density =True, histtype = 'stepfilled')
xplot = np.linspace(min(x), max(x), 100)

# plt.plot(xplot, gaussian(xplot, mean[0], mean[1], mean[2]) , color="darkorange", linewidth=3, alpha=0.6)

plt.plot(xplot, gaussian(xplot, mean[0], mean[1], mean[2]) + gaussian(xplot, mean[3], mean[4], mean[5])
         + gaussian(xplot, mean[6], mean[7], mean[8]), color="darkorange", linewidth=3, alpha=0.6)
plt.plot(xplot, gaussian(xplot, mean[0], mean[1], mean[2])  , color="green", linestyle='dashed', linewidth=3, alpha=0.6)
plt.plot(xplot, gaussian(xplot, mean[3], mean[4], mean[5])  , color="red", linestyle='dashed', linewidth=3, alpha=0.6)
plt.plot(xplot, gaussian(xplot, mean[6], mean[7], mean[8]) , color='black', linestyle='dashed', linewidth=3, alpha=0.6)


# plt.axvline(mean[0],linestyle='dashed',color='orange')
# plt.axvline(mean[3],linestyle='dashed',color='orange')
plt.text(min(x),max(h[0]),'$\mu_{1}=%.3f$'%(mean[0]),color='green')
plt.text(min(x),max(h[0]-0.01),'$\sigma_{1}=%.3f$'%(mean[1]),color='green')
plt.text(min(x),max(h[0]-0.02),'$amp_{1}=%.3f$'%(mean[2]),color='green')
plt.text(max(x)/2,max(h[0]),'$\mu_{2}=%.3f$'%(mean[3]),color='red')
plt.text(max(x)/2,max(h[0]-0.01),'$\sigma_{2}=%.3f$'%(mean[4]),color='red')
plt.text(max(x)/2,max(h[0]-0.02),'$amp_{2}=%.3f$'%(mean[5]),color='red')
plt.text(max(x)/2,max(h[0]-0.03),'$\mu_{3}=%.3f$'%(mean[6]))
plt.text(max(x)/2,max(h[0]-0.04),'$\sigma_{3}=%.3f$'%(mean[7]))
plt.text(max(x)/2,max(h[0]-0.045),'$amp_{3}=%.3f$'%(mean[8]))

plt.text(min(x),max(h[0]/2)-0.01,'$logz=%.0f$'%(results['logz'][-1]),color='b')
if accu<10:
    plt.text(min(x),max(h[0]/2)-0.005,'$\sigma_{vx}<%.1f\ mas\ a^{-1}$'%(accu),color='b')
plt.text(max(x)/2,max(h[0]/2)-0.020,'$nbins=%s$'%(nbins),color='b')
if chip=='both':
    plt.text(max(x)/2,max(h[0]-0.05),'$field%s,\ c%s$'%('All',chip),color='b')
else:
    plt.text(max(x)/2,max(h[0]-0.05),'$field%s,\ c%s$'%(field,chip),color='b')
plt.ylabel('N')
# plt.xlabel(r'$\mu_{l}$ (Km s$^{-1}$)') 
plt.xlabel('v$_{x}$ (mas yr$^{-1}$), IDL') 


#%%
results = sampler.results
print(results['logz'][-1])


# In[ ]:



