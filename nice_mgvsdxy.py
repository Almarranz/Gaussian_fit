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
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})
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
rcParams.update({'font.size': 15})

#%%
data='/Users/amartinez/Desktop/for_Rainer/same_pix_scale/'
zoc = np.loadtxt(data +'Zoc_c3_commons_w_GNS.txt')# a ,d , m, dm, f, df,x,y,dx,dy,x_dis,y_dis. X and Y are the correspondig coorinates wit GNS, They are not the original ones!!!!
gns= np.loadtxt(data +'GNS_commons_w_Zoc_c3.txt') ## x_gns, dx_gns, y_gns, dy_gns, raH, draH, decH, ddecH, mJ, dmJ, mH, dmH, mK, dmK, H-Ks

gns_dxy=np.sqrt((gns[:,1])**2+(gns[:,3])**2)
zoc_dxy=np.sqrt((zoc[:,8])**2+(zoc[:,9])**2)

fig, ax =plt.subplots(1,1,figsize=(10,10))
p1=ax.scatter(gns[:,10],gns_dxy*1000*0.106,color='red',s=30,marker='s')
p2=ax.scatter(gns[:,10],zoc_dxy*1000*0.106,color='blue',s=30, marker='D')
ax.grid()
ax.legend(['GNS', 'Zoc'],fontsize=20,markerscale=3,shadow=True,loc=2)
p1.set_alpha(0.3)# toset the alpha of the legend differently, we make the plot and then the legeng (default alpha =1), and then change the alpga for the plots
p2.set_alpha(0.3)
plt.ylabel(r'$\sigma_{xy}$ (mas)')
ax.set_ylim(0,10)
# plt.xlabel(r'$\mu_{l}$ (Km s$^{-1}$)') 
plt.xlabel(r'[H]')#\ Chip \ %s$'%(chip)) 


















