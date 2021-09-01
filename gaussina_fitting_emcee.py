#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import dynesty


# In[2]:


band='H'
folder='im_jitter_NOgains/'
exptime=10
data='/Users/amartinez/Desktop/PhD/python/Gaussian_fit/'
tmp='/Users/amartinez/Desktop/PhD/HAWK/The_Brick/photometry/054_'+band+'/dit_'+str(exptime)+'/'+folder+'tmp_bs/'


# In[3]:


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


# In[4]:


lst=np.loadtxt(tmp+'lst.txt')
accu=10
nbins=21
chip=3
v_x,v_y,dvx,dvy=np.loadtxt(data+'arcsec_vx_vy_chip3.txt',unpack=True)
# select=np.where((dvx<accu)&(dvy<accu))
select=np.where((dvx<accu))
v_x=v_x[select]
fig,ax=plt.subplots(1,1)
h=ax.hist(v_x,bins=nbins,edgecolor='black',linewidth=2,density=True)
x=[h[1][i]+(h[1][1]-h[1][0])/2 for i in range(len(h[0]))]#middle value for each bin
ax.axvline(np.mean(v_x), color='r', linestyle='dashed', linewidth=3)
ax.legend(['Chip=%s, %s, mean= %.2f, std=%.2f'
              %(chip,len(v_x),np.mean(v_x),np.std(v_x))],fontsize=12,markerscale=0,shadow=True,loc=3,handlelength=-0.0)
y=h[0]#height for each bin
#yerr = y*0.05
#yerr = y*0.01
yerr=0.0001
y += yerr
ax.scatter(x,y,color='g',zorder=3)


# In[5]:


def gaussian(x, mu, sig, amp):
    return amp * (1 / (sig * (np.sqrt(2 * np.pi)))) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


# In[6]:


def model(theta):
     mu1, sigma1, amp1,mu2,sigma2,amp2 = theta
    #model =gaussian(x,mu0,sig0,amp0)
    
     return amp1 * (1 / (sigma1 * (np.sqrt(2 * np.pi)))) * np.exp(-np.power(x - mu1, 2.) / (2 * np.power(sigma1, 2.))) + amp2 * (1 / (sigma2 * (np.sqrt(2 * np.pi)))) * np.exp(-np.power(x - mu2, 2.) / (2 * np.power(sigma2, 2.)))
#      return gaussian(x, mu1, sigma1, amp1)+gaussian(x,mu2,sigma2,amp2)
# def model(theta, x):
#     h,mu, sig= theta
#     model = h*np.exp(-0.5*((x-mu)/(sig))**2)
#     return model


# In[7]:


def lnlike(theta,x,y,yerr):
    
    mu1, sigma1, amp1,mu2,sigma2,amp2 = theta
    #model = gaussian(x, mu0, sigma0, amp0)

    return -0.5 * np.sum(((y - model(theta))/yerr) ** 2)


# In[8]:


a=4
b=5
c=1
def lnprior(theta):
    mu1, sigma1, amp1,mu2,sigma2,amp2 = theta
    if  -a< mu1 < a and 0 < sigma1 < b and 0< amp1 < c and -a< mu2 < a and 0 < sigma2 < b and 0< amp2 < c :
        return 0.0
    return -np.inf


# In[9]:


def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)


# In[10]:


data = (x, y,yerr)
nwalkers = 400
niter = 1000
initial = np.array([0, 2,1,0,2,1])
ndim = len(initial)
p0 = [np.array(initial) + 1e-7 * np.random.randn(ndim) for i in range(nwalkers)]


# In[11]:


def main(p0,nwalkers,niter,ndim,lnprob,data):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

    print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc(p0, 100)
    sampler.reset()

    print("Running production...")
    pos, prob, state = sampler.run_mcmc(p0, niter)

    return sampler, pos, prob, state


# In[12]:


sampler, pos, prob, state = main(p0,nwalkers,niter,ndim,lnprob,data)


# In[13]:


def plotter(sampler,age=x,T=y):
    plt.ion()
    plt.plot(age,T,label='Change in T')
    samples = sampler.flatchain
    for theta in samples[np.random.randint(len(samples), size=2)]:
        plt.plot(age, model(theta, age), color="r", alpha=0.1)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xlabel('Years ago')
    plt.ylabel(r'$\Delta$ T (degrees)')
    plt.legend()
    plt.show()


# In[14]:


samples = sampler.flatchain
fig,ax=plt.subplots(1,1)
ax.plot(x,y)
for theta in samples[np.random.randint(len(samples), size=1000)]:
        ax.plot(x, model(theta), color="r", alpha=0.1)


# In[15]:


samples = sampler.flatchain
samples[np.argmax(sampler.flatlnprobability)]


# In[16]:


samples = sampler.flatchain

theta_max  = samples[np.argmax(sampler.flatlnprobability)]
best_fit_model = model(theta_max)
plt.plot(x,y)
plt.plot(x,best_fit_model,label='Highest Likelihood Model')
plt.show()
print ('Theta max: ',theta_max)


# In[17]:


theta_max[0]


# In[18]:


def gaussian(x, mu, sig, amp):
    return amp * (1 / (sig * (np.sqrt(2 * np.pi)))) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


# In[19]:


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


h=plt.hist(v_x, bins= nbins, color='darkblue', alpha = 0.6, density =True, histtype = 'stepfilled')
xplot = np.linspace(min(x), max(x), 100)

# plt.plot(xplot, gaussian(xplot, mean[0], mean[1], mean[2]) , color="darkorange", linewidth=3, alpha=0.6)

plt.plot(xplot, gaussian(xplot, theta_max[0], theta_max[1], theta_max[2]) + gaussian(xplot, theta_max[3], theta_max[4], theta_max[5]), color="darkorange", linewidth=3, alpha=0.6)
plt.plot(xplot, gaussian(xplot, theta_max[0], theta_max[1], theta_max[2])  , color="red", linestyle='dashed', linewidth=3, alpha=0.6)
plt.plot(xplot, gaussian(xplot, theta_max[3], theta_max[4], theta_max[5])  , color="darkorange", linestyle='dashed', linewidth=3, alpha=0.6)

# plt.axvline(mean[0],linestyle='dashed',color='orange')
# plt.axvline(mean[3],linestyle='dashed',color='orange')
plt.text(min(x),max(h[0]),'$\mu_{1}=%.3f$'%(theta_max[0]),color='red')
plt.text(min(x),max(h[0]-0.01),'$\sigma_{1}=%.3f$'%(theta_max[1]),color='red')
plt.text(max(x)/2,max(h[0]),'$\mu_{2}=%.3f$'%(theta_max[3]))
# plt.text(min(x),max(h[0]-0.02),'$logz=%.0f$'%(results['logz'][-1]))
plt.text(max(x)/2,max(h[0]-0.02),'$nbins=%s$'%(nbins))
plt.text(max(x)/2,max(h[0]-0.01),'$\sigma_{2}=%.3f$'%(theta_max[4]))
plt.text(max(x)/2,max(h[0]-0.03),'$list = %.0f$'%(lst))
plt.ylabel('N')
# plt.xlabel(r'$\mu_{l}$ (Km s$^{-1}$)') 
plt.xlabel('v$_{x}$ (mas yr$^{-1}$)') 


# In[20]:


labels = ['mu1','sigma1','amp1','mu2','sigma2','amp3']
fig = corner.corner(samples,show_titles=True,labels=labels,plot_datapoints=True,plot_contours=True,alpha=0.01)#,quantiles=[0.16, 0.5, 0.84])

