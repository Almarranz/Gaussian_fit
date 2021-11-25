#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 15:41:32 2021

@author: amartinez
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import dynesty
from astropy.stats import sigma_clip
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})
# for Palatino and other serif fonts use:
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = 'dejavuserif'
plt.rcParams['text.usetex'] = False
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})
plt.xlabel(r'$\mathrm{v_{b} (mas\ a^{-1})}$')
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

#%%
# data='/Users/amartinez/Desktop/for_Rainer/same_pix_scale/'
# zoc = np.loadtxt(data +'Zoc_c3_commons_w_GNS.txt')# a ,d , m, dm, f, df,x,y,dx,dy,x_dis,y_dis. X and Y are the correspondig coorinates wit GNS, They are not the original ones!!!!
# gns= np.loadtxt(data +'GNS_commons_w_Zoc_c3.txt') ## x_gns, dx_gns, y_gns, dy_gns, raH, draH, decH, ddecH, mJ, dmJ, mH, dmH, mK, dmK, H-Ks

# gns_dxy=np.sqrt((gns[:,1])**2+(gns[:,3])**2)
# zoc_dxy=np.sqrt((zoc[:,8])**2+(zoc[:,9])**2)

# fig, ax =plt.subplots(1,1,figsize=(10,10))
# p1=ax.scatter(gns[:,10],gns_dxy*1000*0.106,color='red',s=30,marker='s')
# p2=ax.scatter(gns[:,10],zoc_dxy*1000*0.106,color='blue',s=30, marker='D')
# ax.grid()
# ax.legend(['GNS', 'Zoc'],fontsize=20,markerscale=3,shadow=True,loc=2)
# p1.set_alpha(0.3)# toset the alpha of the legend differently, we make the plot and then the legeng (default alpha =1), and then change the alpga for the plots
# p2.set_alpha(0.3)
# plt.ylabel(r'$\sigma_{xy}$ (mas)')
# ax.set_ylim(0,15)
# ax.axhline(np.mean(gns_dxy)*1000*0.106, color='r', linestyle='dashed', linewidth=3)
# ax.axhline(np.mean(zoc_dxy)*1000*0.106, color='blue', linestyle='dashed', linewidth=3)
# plt.text(13,np.mean(gns_dxy)*1000*0.106 +0.15,r'mean=%.3f'%(np.mean(gns_dxy)*1000*0.106),color='red',fontsize=20)
# plt.text(13,np.mean(zoc_dxy)*1000*0.106 +0.15,r'$\sigma_{2}$=%.3f'%(np.mean(zoc_dxy)*1000*0.106),color='blue',fontsize=20)
# # plt.xlabel(r'$\mu_{l}$ (Km s$^{-1}$)') 
# plt.xlabel(r'[H]')#\ Chip \ %s$'%(chip)) 
#%%
band='H'
exptime=10
#chip=1
folder='im_jitter_NOgains/'
results='/Users/amartinez/Desktop/PhD/HAWK/The_Brick/photometry/054_'+band+'/dit_'+str(exptime)+'/'+folder+'/results_bs/'
pruebas='/Users/amartinez/Desktop/PhD/HAWK/The_Brick/photometry/pruebas/'

name='NPL_054'
chip=3
ra,dec,x_mean,dx,y_mean,dy,mag,dmag,l,b=np.loadtxt(results+name+'_chip%s.txt'%(chip),unpack=True)#header='ra,dec,x_mean,dx,y_mean,dy,mag,dmag,l,b'
dxy=np.sqrt((dx)**2+(dy)**2)
fig, ax =plt.subplots(1,1,figsize=(10,10))
ax.scatter(mag,dxy*1000,color='k',s=5, marker='.')
ax.grid()
plt.ylabel(r'$\mathrm{\sigma_{xy} (mas)}$')
ax.set_ylim(0,14)
ax.set_xlim(12,19)

# ax.axhline(np.mean(gns_dxy)*1000*0.106, color='r', linestyle='dashed', linewidth=3)
# ax.axhline(np.mean(zoc_dxy)*1000*0.106, color='blue', linestyle='dashed', linewidth=3)
# plt.text(13,np.mean(gns_dxy)*1000*0.106 +0.15,r'mean=%.3f'%(np.mean(gns_dxy)*1000*0.106),color='red',fontsize=20)
# plt.text(13,np.mean(zoc_dxy)*1000*0.106 +0.15,r'$\sigma_{2}$=%.3f'%(np.mean(zoc_dxy)*1000*0.106),color='blue',fontsize=20)
# plt.xlabel(r'$\mu_{l}$ (Km s$^{-1}$)') 
plt.xlabel(r'[H]')#\ Chip \ %s$'%(chip)) 

#%%
rcParams.update({'font.size': 30})
x_in=np.loadtxt(pruebas+'dvx_mag_IN.txt')#header='mh_all,dvx_all')
no_x_in=np.loadtxt(pruebas+'NO_dvx_mag_IN.txt')#header='mh_all,dvx_all')

x_out=np.loadtxt(pruebas+'dvx_mag_OUT.txt')#header='mh_all,dvx_all')
no_x_out=np.loadtxt(pruebas+'NO_dvx_mag_OUT.txt')#header='mh_all,dvx_all')


ejes=[x_in[0,:],x_out[0,:]]
absc=[x_in[1,:],x_out[1,:]]

ejes_no=[no_x_in[0,:],no_x_out[0,:]]
absc_no=[no_x_in[1,:],no_x_out[1,:]]

zones=['Zone A', 'Zone B']

accu=[2,2]
fig, ax=plt.subplots(1,2,figsize=(20,10))
for i in range(len(ejes)):
    ax[i].scatter(ejes[i],absc[i],color='k',alpha=0.7,s=5)
    # ax[i].scatter(mh_all[no_sel],ejes_accu[i],color='red',alpha=0.7,s=5)
    ax[i].scatter(ejes_no[i],absc_no[i],color='red',alpha=1,s=30)
    ax[i].axhline(accu[i], color='r', linestyle='dashed', linewidth=3)
    # ax[i].axvline(max_M, color='r', linestyle='dashed', linewidth=3)
    ax[i].set_xlim(12,19)
    ax[i].set_ylim(0,8)
    ax[i].set_xlabel('[H]',fontsize=40)
    ax[i].set_ylabel(r'$\mathrm{\sigma_{\vec {vx}}(mas\ a^{-1})}$',fontsize=30)
    ax[i].legend(['%s'%(zones[i])],fontsize=30,markerscale=0,shadow=True,loc=2,handlelength=-0.0)






