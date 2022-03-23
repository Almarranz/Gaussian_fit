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
rcParams.update({'font.size': 20})

band='H'
exptime=10
#chip=1
folder='im_jitter_NOgains/'
pruebas='/Users/amartinez/Desktop/PhD/HAWK/The_Brick/photometry/pruebas/'
data='/Users/amartinez/Desktop/PhD/HAWK/The_Brick/photometry/extintion/'
gaussian='/Users/amartinez/Desktop/PhD/HAWK/The_Brick/photometry/058_'+band+'/dit_'+str(exptime)+'/'+folder+'Gaussian_fit/'
field ='c'#cfor control and f for target
if field == 'c':
   x_coor,y_coor,vx,vy,H,K, AKs= np.loadtxt(data +'control_extintion.txt',unpack=True)
   dvx,dvy=np.loadtxt(gaussian+'%s_comparison_field%s_chip%s_degree%s.txt'%('Z1',16,3,2),unpack=True,usecols=(6,7))#arg,decg,v_x,v_y,mk,mh,dvx,dvy    
elif field == 't':
     x_coor,y_coor,vx,vy,H,K, AKs= np.loadtxt(data +'target_extintion.txt',unpack=True)
print(np.std(dvx))
vx_lim=np.where((vx < 120) & (vx > -120)) 
vx=vx[vx_lim]
vx=vx*-1
AKs=AKs[vx_lim]
# =============================================================================
# x_c,y_c,vx_c,vy_c,H_c,K_c, AKs_c= np.loadtxt(data +'control_extintion.txt',unpack=True)
# x_t,y_t,vx_t,vy_t,H_t,K_t, AKs_t= np.loadtxt(data +'target_extintion.txt',unpack=True)
# =============================================================================




# =============================================================================
# fig, ax =plt.subplots(1,1,figsize=(8,8))
# ax.scatter(vx,AKs,color='k',s=15, marker='.')
# ax.grid()
# plt.ylabel(r'$\mathrm{A_{Ks}}$')
# =============================================================================
# plt.hist(AKs)
# ax.set_ylim(0,14)
# ax.set_xlim(12,19)

# ax.axhline(np.mean(gns_dxy)*1000*0.106, color='r', linestyle='dashed', linewidth=3)
# ax.axhline(np.mean(zoc_dxy)*1000*0.106, color='blue', linestyle='dashed', linewidth=3)
# plt.text(13,np.mean(gns_dxy)*1000*0.106 +0.15,r'mean=%.3f'%(np.mean(gns_dxy)*1000*0.106),color='red',fontsize=20)
# plt.text(13,np.mean(zoc_dxy)*1000*0.106 +0.15,r'$\sigma_{2}$=%.3f'%(np.mean(zoc_dxy)*1000*0.106),color='blue',fontsize=20)
# plt.xlabel(r'$\mu_{l}$ (Km s$^{-1}$)') 
# plt.xlabel(r'$\mathrm{\mu_{l}(mas a^{-1})}$')#\ Chip \ %s$'%(chip)) 

#%%
fig, ax =plt.subplots(1,1,figsize=(8,8))
bi=np.arange(-12,15,3)
n,bins,_=ax.hist(vx,bins=bi)
all_AKs=[]
all_dvx=[]
dKs=[]
dKs1=[]# here I gonna use the uncertainty for Ks = 0.003

for i in range(len(bins)-1):
    m_AKs=[]
    m_dvx=[]
    for j in range(len(vx)):
        if vx[j]>=bins[i] and vx[j]<=bins[i+1]:
             m_AKs.append(AKs[j])  
             m_dvx.append(dvx[j])
        
    # print(bins[i],bins[i+1])
        
# =============================================================================
#         try:
#             if vx[j]>=bins[i] and vx[j]<=bins[i+1]:
#                 m_AKs.append(AKs[j])
#                 # print(vx[j],AKs[j])
#         except:
#             if vx[j]>=bins[i]:
#                 m_AKs.append(AKs[j])
#                 print(vx[j],'FALLO')
# =============================================================================
    
    if len(m_AKs)>int(round(len(vx)*-0.000000001)):
        print('This is the length:',len(m_AKs))
        all_AKs.append(np.mean(m_AKs))
        all_dvx.append(np.std(m_dvx)/np.sqrt(len(m_AKs)))
        dKs.append(np.std(m_AKs)/np.sqrt(len(m_AKs)))
        dKs1.append(0.03/np.sqrt(len(m_AKs)))

        # all_AKs.append(np.median(m_AKs))
    else:
        all_AKs.append(0)
        dKs.append(np.std(m_AKs)/np.sqrt(len(m_AKs)))
        dKs1.append(0.03/np.sqrt(len(m_AKs)))
        print(30*'#')
    print(bins[i],bins[i+1],all_AKs[i],dKs[i])
fig, ax =plt.subplots(1,1,figsize=(8,8))
# ax.scatter(vx,AKs,color='k',s=15, marker='.',alpha=0.5)
ax.scatter(bi[:-1],all_AKs,s=3,color='red',marker='*',zorder=3,alpha=1)
ax.errorbar(bi[:-1],all_AKs,dKs,all_dvx,color='darkblue',fmt='none',capsize=3,alpha=0.5)
ax.set_ylim(1.4,3)
ax.set_xlim(-10.5,10.5)
ax.grid()
plt.ylabel(r'$\mathrm{A_{Ks}}$')
plt.xlabel(r'$\mathrm{\mu_{l}(mas\ yr^{-1})}$')#\ Chip \ %s$'%(chip)) 
plt.gca().invert_xaxis()
if field=='c':
    plt.legend(['Comparison Field'],fontsize=20,markerscale=1,shadow=True,loc=1,handlelength=0.5)
    np.savetxt(data + 'Aks_vs_vx_control.txt',np.array([bi[:-1]+1.5,all_AKs,dKs,dKs1]).T)
if field=='t':
    plt.legend(['Brick Field'],fontsize=20,markerscale=1,shadow=True,loc=1,handlelength=0.5)
    np.savetxt(data + 'Aks_vs_vx_target.txt',np.array([bi[:-1]+1.5,all_AKs,dKs,dKs1]).T)

#%%
rcParams.update({'font.size': 20})
fig, ax =plt.subplots(1,1,figsize=(8,8))
# ax.scatter(vx,AKs,color='k',s=15, marker='.',alpha=0.5)
bi_c,AK_c,dK_c,dK1_c=np.loadtxt(data+'Aks_vs_vx_control.txt',unpack=True)
bi_t,AK_t,dK_t,dK1_t=np.loadtxt(data+'Aks_vs_vx_target.txt',unpack=True)

a=0.0
b=-0.0

ax.scatter(bi_c+a,AK_c,s=100,color='k',marker='s',zorder=3,alpha=1, label= 'Comparison field')
ax.errorbar(bi_c+a,AK_c,dK_c,np.array(all_dvx),color='k',fmt='none',capsize=3,alpha=1)
plt.legend(['Comparison Field'],fontsize=20,markerscale=0,shadow=True,loc=2,handlelength=-0.8)

# =============================================================================
# ax.scatter(bi_t+b,AK_t,s=100,color='red',marker='s',zorder=3,alpha=1, label='Brick field')
# ax.errorbar(bi_t+b,AK_t,dK_t,color='red',fmt='none',capsize=3,alpha=1,zorder=3)
# ax.legend(fontsize=20,markerscale=1,shadow=True,loc=1,handlelength=0.5)
# =============================================================================

# plt.legend(['Comparison Field'],fontsize=20,markerscale=1,shadow=True,loc=1,handlelength=0.5)
ax.set_ylim(1.6,2.2)
ax.set_xlim(-10,10)
ax.set_xticks( [10,-10,0,5,-5], minor=False)
ax.set_yticks( [1.6,1.7,1.8,1.9,2.0,2.1,2.2], minor=False)

# ax.grid()
plt.ylabel(r'$\mathrm{A_{Ks}}$')
plt.xlabel(r'$\mathrm{\mu_{l}(mas\ yr^{-1})}$')#\ Chip \ %s$'%(chip)) 
plt.gca().invert_xaxis()
pics='/Users/amartinez/Desktop/PhD/My_papers/brick/corrected_by_language/'
plt.savefig(pics + 'extintion_vx_comparison.png', dpi=300,bbox_inches='tight')

#%%





#%%
# =============================================================================
# fig, ax =plt.subplots(1,1,figsize=(8,8))
# hb = ax.hexbin(vx, AKs, gridsize=8, cmap='rainbow',bins='log')
# cb = fig.colorbar(hb, ax=ax)
# plt.gca().invert_xaxis()
# # #ax.scatter(v_x,v_y,s=1,color='black')
# # #fig.add_colorbar()
# # #fig.colorbar.set_font(size=font_size)
# # #fig.colorbar.set_axis_label_text('Number of objects')
# # #fig.colorbar.set_axis_label_font(size=14)
# =============================================================================
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
    ax[i].scatter(ejes_no[i],absc_no[i],color='red',alpha=0.7,s=25)
    ax[i].axhline(accu, color='r', linestyle='dashed', linewidth=3)
    # ax[i].axvline(max_M, color='r', linestyle='dashed', linewidth=3)
    ax[i].set_xlim(12,19)
    ax[i].set_ylim(0,8)
    ax[i].set_xlabel('[H]',fontsize=40)
    ax[i].set_ylabel(r'$\mathrm{\sigma_{\vec {vx}}(mas)}$',fontsize=40)
    ax[i].legend(['%s'%(zones[i])],fontsize=40,markerscale=0,shadow=True,loc=1,handlelength=-0.0)

#%%

 # =====================RINER nice plots========================================================
# hb = ax.hexbin(v_x, v_y, gridsize=55, cmap='rainbow',)#bins='log')
# #cb = fig.colorbar(hb, ax=ax)
# #ax.scatter(v_x,v_y,s=1,color='black')
# #fig.add_colorbar()
# #fig.colorbar.set_font(size=font_size)
# #fig.colorbar.set_axis_label_text('Number of objects')
# #fig.colorbar.set_axis_label_font(size=14)
# 
# # ~ cb = fig.colorbar(hb, ax=ax)
# 
# from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
# from matplotlib.transforms import Bbox
# 
# 
# class RemainderFixed(axes_size.Scaled):
#     def __init__(self, xsizes, ysizes, divider):
#         self.xsizes =xsizes
#         self.ysizes =ysizes
#         self.div = divider
# 
#     def get_size(self, renderer):
#         xrel, xabs = axes_size.AddList(self.xsizes).get_size(renderer)
#         yrel, yabs = axes_size.AddList(self.ysizes).get_size(renderer)
#         bb = Bbox.from_bounds(*self.div.get_position()).transformed(self.div._fig.transFigure)
#         w = bb.width/self.div._fig.dpi - xabs
#         h = bb.height/self.div._fig.dpi - yabs
#         return 0, min([w,h])
# 
# 
# def make_square_axes_with_colorbar(ax, size=0.1, pad=0.1):
#     """ Make an axes square, add a colorbar axes next to it, 
#         Parameters: size: Size of colorbar axes in inches
#                     pad : Padding between axes and cbar in inches
#         Returns: colorbar axes
#     """
#     divider = make_axes_locatable(ax)
#     margin_size = axes_size.Fixed(size)
#     pad_size = axes_size.Fixed(pad)
#     xsizes = [pad_size, margin_size]
#     yhax = divider.append_axes("right", size=margin_size, pad=pad_size)
#     divider.set_horizontal([RemainderFixed(xsizes, [], divider)] + xsizes)
#     divider.set_vertical([RemainderFixed(xsizes, [], divider)])
#     return yhax
# 
# 
# cax = make_square_axes_with_colorbar(ax, size=0.09, pad=0.1)
# cbar = fig.colorbar(hb, cax=cax)
# 
# cbar.set_label('Number of Objects', rotation=90)
# 
# 
# 
# plt.savefig('GNS_PMs_VPD_GC.png',dpi=200,bbox_inches='tight')
# plt.show()
# 
# =============================================================================    
    
    
    
    
    
    
    
    
    
    
    




