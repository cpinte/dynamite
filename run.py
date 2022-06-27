import os
import measure_height as mh
import my_casa_cube as casa

# HD163296
#data = casa.Cube('./HD163296_CO.pbcor.fits')
#mh.Surface(data, PA=133, inc=47, x_c=480, y_c=477, v_syst=5.7, sigma=5) 

# HD169142
#data = casa.Cube('./HD169142_12co_contsub_r0.5_uv0.08_clipped.fits')
#mh.Surface(data, PA=5, inc=13, x_c=400, y_c=400, v_syst=6.9, sigma=10)

# HD169142 non-clipped cube
#data = casa.Cube('./HD169142_12co_contsub_r1.5_uv0.1.fits')
#mh.Surface(data, PA=5, inc=13, x_c=900, y_c=900, v_syst=6.9, sigma=10) 

# MWC758
#data = casa.Cube('./MWC_758_TM1_12CO_selfcal_v4.9-6.9_dv0.1kms.image.pbcor.fits')
#mh.Surface(data, PA=65, inc=21, x_c=500, y_c=500, v_syst=5.9, sigma=5) 

# CQ Tau exoALMA
#data = casa.Cube('./CQ_Tau_12CO_robust0.5_width0.1kms_threshold4.0sigma.clean.image.fits')
#mh.Surface(data, PA=55, inc=35, x_c=513, y_c=514, v_syst=6.17, sigma=5)

# DM Tau exoALMA
#data = casa.Cube('./DM_Tau_12CO_robust0.5_width0.1kms_threshold4.0sigma.clean.image.fits')
#mh.Surface(data, PA=158, inc=35, x_c=513, y_c=514, v_syst=6.1, sigma=5)

###### Synthetic cubes ###########

#test1, co_abs_zr_gt0.0
data = casa.Cube('./co_abs_zr_gt0.0_conv.fits')
mh.Surface(data, PA=270, inc=45, x_c=158, y_c=158, v_syst=5.9, sigma=10)

#test2; co_abs_zr_gt0.3
#data = casa.Cube('./co_abs_zr_gt0.3_conv.fits')
#mh.Surface(data, PA=270, inc=45, x_c=158, y_c=158, v_syst=5.9, sigma=7)


