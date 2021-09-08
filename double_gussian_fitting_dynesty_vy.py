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


# In[3]:
band='H'
folder='im_jitter_NOgains/'
exptime=10
data='/Users/amartinez/Desktop/PhD/python/Gaussian_fit/'
tmp='/Users/amartinez/Desktop/PhD/HAWK/The_Brick/photometry/054_'+band+'/dit_'+str(exptime)+'/'+folder+'tmp_bs/'

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
rcParams.update({'font.size': 30})
rcParams.update({'figure.figsize':(10,5)})
rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})


# In[5]:


chip=3

nbins=19
accu=1000
in_brick=1
if in_brick==1:
    if chip =='both':
        v_x2,v_y2,dvx2,dvy2=np.loadtxt(data+'arcsec_vx_vy_chip2.txt',unpack=True)
        v_x3,v_y3,dvx3,dvy3=np.loadtxt(data+'arcsec_vx_vy_chip3.txt',unpack=True)
        v_x=np.r_[v_x2,v_x3]
        v_y=np.r_[v_y2,v_y3]
        dvx=np.r_[dvx2,dvx3]
        dvy=np.r_[dvy2,dvy3]
    elif chip==2 or chip==3:
        lst=np.loadtxt(tmp+'lst_chip%s.txt'%(chip))
        v_x,v_y,dvx,dvy=np.loadtxt(data+'arcsec_vx_vy_chip%s.txt'%(chip),unpack=True)
elif in_brick==0:
        lst=3
        v_x,v_y,dvx,dvy=np.loadtxt(data+'arcsec_vx_vy_chip%s_out_Brick.txt'%(chip),unpack=True)        


select=np.where((dvy<accu))
v_y=v_y[select]
fig,ax=plt.subplots(1,1)
h=ax.hist(v_y,bins=nbins,edgecolor='black',linewidth=2,density=True)
x=[h[1][i]+(h[1][1]-h[1][0])/2 for i in range(len(h[0]))]#middle value for each bin
ax.axvline(np.mean(v_y), color='r', linestyle='dashed', linewidth=3)
ax.legend(['Chip=%s, %s, mean= %.2f, std=%.2f'
              %(chip,len(v_y),np.mean(v_y),np.std(v_y))],fontsize=12,markerscale=0,shadow=True,loc=3,handlelength=-0.0)
y=h[0]#height for each bin
#yerr = y*0.05
#yerr = y*0.01
yerr=0.0001
y += yerr
ax.scatter(x,y,color='g',zorder=3)


# In[6]:


# for i in range(len(yerr)):
#     if yerr[i] ==0:
#         yerr[i]=np.mean(yerr[i-1],yerr[i+1])


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
    mu1 = 2*umu1-1  # scale and shift to [-3., 3.)
    sigma1 = (usigma1)*4
    amp1 = uamp1*3
    
    mu2 =2*umu2-1 # scale and shift to [-3., 3.)
    sigma2 = (usigma2)*8
    amp2 = uamp2*4

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
#                               fig=plt.subplots(6, 6, figsize=(28, 28)))

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
plt.text(max(x)/2,max(h[0]),'$\mu_{2}=%.3f$'%(mean[3]))
plt.text(min(x),max(h[0]-0.02),'$logz=%.0f$'%(results['logz'][-1]),color='b')
plt.text(max(x)/2,max(h[0]-0.02),'$nbins=%s$'%(nbins),color='b')
plt.text(max(x)/2,max(h[0]-0.01),'$\sigma_{2}=%.3f$'%(mean[4]))
if (chip==2 or chip==3) and in_brick==1:
    plt.text(max(x)/2,max(h[0]-0.03),'$list = %.0f$'%(lst))
elif in_brick==0:
    plt.text(max(x)/2,max(h[0]-0.03),'$list = %s$'%('out Brick'))
plt.ylabel('$N$')
# plt.xlabel(r'$\mu_{l}$ (Km s$^{-1}$)') 
plt.xlabel('$v_{y} (mas\ yr^{-1}), Chip %s$'%(chip)) 

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
