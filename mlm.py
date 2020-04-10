#### MLM : Molecular Layer Mapper

import casa_cube as casa
import measure_height
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate
from colorama import Fore, Style
import os

# enter inclination (inc) and position angle (PA) in degrees.

#Sz (y,x) = 750, 752  , WaOph6 (y,x) = 720,719
#sources systemic velocity:
# IM Lupi = 48 inc, 143 PA, 4.5 km/s , 161 pc.

#sources = ['AS209', 'HD163296', 'SR4', 'Sz129', 'WaOph6', 'DoAR25', 'Elias27', 'GWLup', 'HD142666', 'HD143006']
#inc_list = [35, 47, 22, 34, 47, 67, 56, 38, 62, 19,]
#PA_list = [87, 133, 18, 151, 174, 111, 119, 38, 162, 169]
#v_syst_list = [4.75, 5.7, 5, 4, 3.85]#, -, -, -, -, -]
#distance_list = [121, 101, 134, 161, 123, 138, 116, 155, 148, 165]

### plotting options
plot_layers = True
plot_continuum = False
plot_rotated_cube = False

### adjust these parameters for each source
source_name = 'IM_Lupi'
isotope = ['CO','13CO','C18O']
inclination = 48                  # in degrees
position_angle = 143              # in degrees    
systemic_velocity = 4.5           # same units as in the cube
distance = 161                    # in parsecs

###############################################################################################################

plot = ['height', 'velocity', 'brightness temperature']
axis = ['r (au)', 'h (au)', 'v (km/s)', 'Tb (K)']
'''
for k in range(3):
    plt.figure(plot[k])
    plt.clf()
    plt.xlabel(axis[0], labelpad=7)
    plt.ylabel(axis[k+1], labelpad=7)
'''
continuum = casa.Cube(f'{source_name}_continuum.fits')

inc = np.radians(inclination)
PA = position_angle
v_syst = systemic_velocity
sigma = 5

for i in range(1):
    
    if i == 0:
        source = casa.Cube(f'{source_name}_{isotope[0]}.fits')
        y_star, x_star = measure_height.star_location(source, continuum, PA=PA, plot=plot_continuum, name=source_name)
        
    source = casa.Cube(f'{source_name}_{isotope[i]}.fits')
    print(f'{source_name}_{isotope[i]}')
    
    n, x, y, T = measure_height.detect_surface(source, PA=PA, plot=plot_rotated_cube, sigma=sigma, y_star=y_star)

    r, h, v, Tb = measure_height.measure_mol_surface(source, n, x, y, T, inc=inc, x_star=x_star, y_star=y_star, v_syst=v_syst, distance=distance)
    
    #measure_height.plotting_mol_surface(r, h, v, Tb, i, isotope)
    
    if plot_layers is True:

        directory = f'{source_name}_layers'

        if not os.path.exists(directory):
            os.mkdir(directory)
        
        nv = source.nv
        for j in range(nv):
            iv = j 
            measure_height.plot_surface(source, n, x, y, T, iv, PA=PA, win=20)
            plt.plot(x_star, y_star, '.', color='white')
            plt.savefig(f'{directory}/{source_name}_channel_{j}.png')
'''           
for k in range(3):
    plt.figure(plot[k])
    plt.legend(loc='best')
    plt.savefig(f'{source_name}_{plot[k]}.png')
    
plt.show()
'''    
