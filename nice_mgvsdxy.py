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


band='H'
exptime=10
#chip=1
folder='im_jitter_NOgains/'
results='/Users/amartinez/Desktop/PhD/HAWK/The_Brick/photometry/054_'+band+'/dit_'+str(exptime)+'/'+folder+'/results_bs/'
pruebas='/Users/amartinez/Desktop/PhD/HAWK/The_Brick/photometry/pruebas/'
GNS_p='/Users/amartinez/Desktop/PhD/HAWK/The_Brick/field16/'

#%%
rcParams.update({'font.size': 20})


#chip=1
folder='im_jitter_NOgains/'
data = '/Users/amartinez/Desktop/PhD/HAWK/The_Brick/photometry/dxy_GNS_vs_ZOC/'


# =============================================================================
# =============================================================================
# =============================================================================

mH,dx,dy,draH,ddecH=np.loadtxt(data+'out_comm_GNS_ZOC.txt',unpack=True)
mH_up,dx_up,dy_up,draH_up,ddecH_up=np.loadtxt(data+'inUP_comm_GNS_ZOC.txt',unpack=True)

dx_all=np.r_[dx,dx_up]
dy_all=np.r_[dy,dy_up]
draH_all=np.r_[draH,draH_up]
ddecH_all=np.r_[ddecH,ddecH_up]
mH_all=np.r_[mH,mH_up]

dxyz=np.sqrt((dx_all)**2+(dy_all)**2)
dxyg=np.sqrt((draH_all)**2+(ddecH_all)**2)



size=20
fig, ax =plt.subplots(1,1,figsize=(10,10))
ax.scatter(mH_all,dxyz,color='k',s=size, marker='o',alpha=0.3)
ax.scatter(mH_all,dxyg*1000,color='red',s=size, marker='o',zorder=3,alpha=0.3)
leg=ax.legend(['D19','D15(GNS)'],fontsize=20,markerscale=2.5,shadow=True,loc=2,handlelength=0.5)
for lh in leg.legendHandles: 
    lh.set_alpha(1)
ax.grid()
plt.ylabel(r'$\mathrm{\sigma_{xy} (mas)}$')
ax.set_ylim(0,12)
ax.set_xlim(12,19)

# ax.axhline(np.mean(gns_dxy)*1000*0.106, color='r', linestyle='dashed', linewidth=3)
# ax.axhline(np.mean(zoc_dxy)*1000*0.106, color='blue', linestyle='dashed', linewidth=3)
# plt.text(13,np.mean(gns_dxy)*1000*0.106 +0.15,r'mean=%.3f'%(np.mean(gns_dxy)*1000*0.106),color='red',fontsize=20)
# plt.text(13,np.mean(zoc_dxy)*1000*0.106 +0.15,r'$\sigma_{2}$=%.3f'%(np.mean(zoc_dxy)*1000*0.106),color='blue',fontsize=20)
# plt.xlabel(r'$\mu_{l}$ (Km s$^{-1}$)') 
plt.xlabel(r'[H]')#\ Chip \ %s$'%(chip)) 

#%%


#%%
size=20

dposz=[dx_all,dy_all]
dposg=[draH_all,ddecH_all]
ejes=['x','y']
fig, ax =plt.subplots(1,2,figsize=(20,10))
al=0.3

for i in range(len(dposg)):
    ax[i].scatter(mH_all,dposz[i],color='k',s=size, marker='o',alpha=al)
    ax[i].scatter(mH_all,dposg[i]*1000,color='red',s=size, marker='o',zorder=3,alpha=al)
    leg=ax[0].legend(['D19','D15(GNS)'],fontsize=20,markerscale=2.5,shadow=True,loc=2,handlelength=0.5)
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
    
    ax[i].set_ylim(0,10)
    ax[i].set_xlim(12,19)
    ax[i].set_xlabel(r'[H]',fontsize=40)#\ Chip \ %s$'%(chip)) 
    ax[i].set_ylabel(r'$\mathrm{\sigma_{%s} (mas)}$'%(ejes[i]),fontsize=40,labelpad=-10)
    
    ax[i].grid()


#%%


#%%


#%%


x_in=np.loadtxt(pruebas+'dvx_mag_IN.txt')#header='mh_all,dvx_all')
no_x_in=np.loadtxt(pruebas+'NO_dvx_mag_IN.txt')#header='mh_all,dvx_all')

x_out=np.loadtxt(pruebas+'dvx_mag_OUT.txt')#header='mh_all,dvx_all')
no_x_out=np.loadtxt(pruebas+'NO_dvx_mag_OUT.txt')#header='mh_all,dvx_all')


ejes=[x_in[0,:],x_out[0,:]]
absc=[x_in[1,:],x_out[1,:]]

ejes_no=[no_x_in[0,:],no_x_out[0,:]]
absc_no=[no_x_in[1,:],no_x_out[1,:]]

zones=['Brick field', 'Comparison field']
rcParams.update({'font.size': 40})
accu=2
fig, ax=plt.subplots(1,2,figsize=(20,10))
for i in range(len(ejes)):
    ax[i].scatter(ejes[i],absc[i],color='k',alpha=0.7,s=5)
    # ax[i].scatter(mh_all[no_sel],ejes_accu[i],color='red',alpha=0.7,s=5)
    ax[i].legend(['%s'%(zones[i])],fontsize=40,markerscale=0.0,shadow=True,loc=1,handlelength=-0.8)
    ax[i].scatter(ejes_no[i],absc_no[i],color='red',alpha=0.7,s=25)
    ax[i].axhline(accu, color='r', linestyle='dashed', linewidth=3)
    # ax[i].axvline(max_M, color='r', linestyle='dashed', linewidth=3)
    ax[i].set_xlim(12,19)
    ax[i].set_ylim(0,4)
    ax[i].set_xlabel('[H]',fontsize=40)
    ax[i].set_ylabel(r'$\mathrm{\sigma_{\vec {vx}}(mas\ yr^{-1})}$',fontsize=40)
    ax[i].grid()
    

#%%

# =============================================================================
# mH_new,dvelc=np.loadtxt(pruebas+'dvx_mag_OUT1.txt',unpack=True)#header='mh_all,dvx_all')
# 
# 
# zones=['Brick field', 'Comparison field']
# rcParams.update({'font.size': 40})
# accu=2
# fig, ax=plt.subplots(1,2,figsize=(20,10))
# 
# ax[0].scatter(mH_new,dvelc,color='k',alpha=0.7,s=5)
# # ax[i].scatter(mh_all[no_sel],ejes_accu[i],color='red',alpha=0.7,s=5)
# ax[0].legend(['Out1'],fontsize=40,markerscale=0.0,shadow=True,loc=1,handlelength=-0.8)
# ax[0].axhline(1.5, color='r', linestyle='dashed', linewidth=3)
# # ax[i].axvline(max_M, color='r', linestyle='dashed', linewidth=3)
# ax[0].set_xlim(12,19)
# ax[0].set_ylim(0,4)
# ax[0].set_xlabel('[H]',fontsize=40)
# ax[0].set_ylabel(r'$\mathrm{\sigma_{\vec {vx}}(mas\ yr^{-1})}$',fontsize=40)
# 
# ax[1].scatter(ejes[1],absc[1],color='k',alpha=0.7,s=5)
# # ax[i].scatter(mh_all[no_sel],ejes_accu[i],color='red',alpha=0.7,s=5)
# ax[1].legend(['Out'],fontsize=40,markerscale=0.0,shadow=True,loc=1,handlelength=-0.8)
# ax[1].axhline(1.5, color='r', linestyle='dashed', linewidth=3)
# # ax[i].axvline(max_M, color='r', linestyle='dashed', linewidth=3)
# ax[1].set_xlim(12,19)
# ax[1].set_ylim(0,4)
# ax[1].set_xlabel('[H]',fontsize=40)
# ax[1].set_ylabel(r'$\mathrm{\sigma_{\vec {vx}}(mas\ yr^{-1})}$',fontsize=40)
# 
# =============================================================================

#%%
brick='/Users/amartinez/Desktop/PhD/My_papers/brick/'
im3 = plt.imread(brick+'1_vy_in_poiss.png')
im4 = plt.imread(brick+'2_vx_in_poiss.png')
im1 = plt.imread(brick+'3_vy_out_poiss.png')
im2 = plt.imread(brick+'4_vx_out_poiss.png')
ims=[[im1,im2],[im3,im4]]
fig, ax = plt.subplots(2,2,figsize=(20,20))
for i in range(2):
    for j in range(2):
        ax[i,j].imshow(ims[i][j])
        ax[i,j].axis('off')
plt.subplots_adjust(wspace=-0.005, hspace=0)
    
# =============================================================================
# brick='/Users/amartinez/Desktop/PhD/My_papers/brick/'
# im3 = plt.imread(brick+'1_vy_in_poiss_unc.png')
# im4 = plt.imread(brick+'2_vx_in_poiss_unc.png')
# im1 = plt.imread(brick+'3_vy_out_poiss_unc.png')
# im2 = plt.imread(brick+'4_vx_out_poiss_unc.png')
# ims=[[im1,im2],[im3,im4]]
# fig, ax = plt.subplots(2,2,figsize=(40,40))
# for i in range(2):
#     for j in range(2):
#         ax[i,j].imshow(ims[i][j])
#         ax[i,j].axis('off')
# plt.subplots_adjust(wspace=-0.005, hspace=0)
# =============================================================================
    
#%%

# =============================================================================
# name='NPL_054'
# chip=3
# ra,dec,x_mean,dx,y_mean,dy,mag,dmag,l,b=np.loadtxt(results+name+'_chip%s.txt'%(chip),unpack=True)#header='ra,dec,x_mean,dx,y_mean,dy,mag,dmag,l,b'
# dxy=np.sqrt((dx)**2+(dy)**2)
# fig, ax =plt.subplots(1,1,figsize=(20,20))
# ax.scatter(mag,dy*1000,color='k',s=5, marker='.')
# ax.grid()
# plt.ylabel(r'$\mathrm{\sigma_{xy} (mas)}$')
# ax.set_ylim(0,14)
# ax.set_xlim(12,19)
# 
# # ax.axhline(np.mean(gns_dxy)*1000*0.106, color='r', linestyle='dashed', linewidth=3)
# # ax.axhline(np.mean(zoc_dxy)*1000*0.106, color='blue', linestyle='dashed', linewidth=3)
# # plt.text(13,np.mean(gns_dxy)*1000*0.106 +0.15,r'mean=%.3f'%(np.mean(gns_dxy)*1000*0.106),color='red',fontsize=20)
# # plt.text(13,np.mean(zoc_dxy)*1000*0.106 +0.15,r'$\sigma_{2}$=%.3f'%(np.mean(zoc_dxy)*1000*0.106),color='blue',fontsize=20)
# # plt.xlabel(r'$\mu_{l}$ (Km s$^{-1}$)') 
# plt.xlabel(r'[H]')#\ Chip \ %s$'%(chip)) 
#     
#     
#     
#     
# =============================================================================
    
    
    
    
    
    
    
    
    




