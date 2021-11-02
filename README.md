# Gaussian_fit
## on branch: NPL058


Gaussian fit for proper motion __ON__ The Brick

1. IDL_double_gussian_fitting_dynesty_vy.py. On vy, the parametres of the Gaussian should be free, the run must coverge by itself to two Gaussian.
Parametres to edit are:
* in_brick=0. This __SHOULD__ be 0 in this script. It means you are working on the Brick area
* fld=[16,3]. Field you want to use
* chip =[1,2,3,4]. Chooses the chips. 
* sm=0.5. Absolute difference in magnitud with GNS
* acu=1.5. Max uncertainty in componet vy and vx velocities.
* step=np.arange(1.5,2.25,0.25). The width of the bins

3.  IDL_triple_gussian_fitting_dynesty_vx.py. Here you have to restric the *sigma and amplitude* parametres of one of the Gaussian to the value previously obtained (on step 1) for the Bulge Gussian. The rest of editable parametres should be equal to step 1. 
___
> ### NOTE



