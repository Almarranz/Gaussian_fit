# Gaussian_fit
## on branch: main


Gaussian fit for proper motion __ON__ The Brick
1. vy_IN_bins_probability.py
On vy, the parametres of the Gaussian should be free, the run must coverge by itself to two Gaussian.
Parametres to edit are:
* in_brick=1. This __SHOULD__ be 1 in this script. It means you are working on the Brick area
* chip ='both'. Could be 3, 2, or both. It refers of the chips tha cover (partially) the brick. Try and choose.
* sm=0.5. Absolute difference in magnitud with GNS
* acu=1.5. Max uncertainty in componet vy and vx velocities.
* step='auto' or step=np.arange(1.5,2.25,0.25). The width of the bins. If *auto* the code decide between Freedman-Diaconis´s rule and Surgard´s. 
2. vx_IN 2G_bins_probability.py. Using two Gaussian the logz is better. Erything sholud run smothlly. If you nned it try restrictin some of the paramtres.
___
3. vx_in_bins_probability.py. This one is with 3 Guassia. The 3rd gaussian doesn´t converget. The results are better with 2 gaussian.


### OLD
1. IDL_double_gussian_fitting_dynesty_vy.py. On vy, the parametres of the Gaussian should be free, the run must coverge by itself to two Gaussian.
Parametres to edit are:
* in_brick=1. This __SHOULD__ be 1 in this script. It means you are working on the Brick area
* chip ='both'. Could be 3, 2, or both. It refers of the chips tha cover (partially) the brick. Try an choose.
* sm=0.5. Absolute difference in magnitud with GNS
* acu=1.5. Max uncertainty in componet vy and vx velocities.
* step=np.arange(1.5,2.25,0.25). The width of the bins

2.  IDL_double_gussian_fitting_dynesty_vx.py. Here you have to restric the *sigma and amplitude* parametres of one of the Gaussian to the value previously obtained (on step 1) for the Bulge Gussian. The rest of editable parametres should be equal to step 1. 
___
> ### NOTE
> In principle you can no use a thrid Gaussian here becouse we are no seeing any star from the far side of the NSD, but you can try with
> *  IDL_triple_gussian_fitting_dynesty_vx.py


