#### MLM : Molecular Layer Mapper

#sources systemic velocity:
# IM Lupi = 48 inc, 143 PA, 4.5 km/s , 161 pc.

import casa_cube as casa
import measure_height
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate
from colorama import Fore, Style
from skimage.transform import resize

# enter inclination (inc) and position angle (PA) in degrees.

### plotting options
plot_layers = False
plot_continuum = False
plot_rotated_cube = False

#Sz (y,x) = 750, 752  , WaOph6 (y,x) = 720,719

DSHARP_list = ['WaOph6']# ['AS209', 'HD163296', 'SR4', 'Sz129', 'WaOph6', 'DoAR25', 'Elias27', 'GWLup', 'HD142666', 'HD143006']
isotope = ['CO','13CO','C18O']
inc_list = [47]# [35, 47, 22, 34, 47, 67, 56, 38, 62, 19,]
PA_list = [174]# [87, 133, 18, 151, 174, 111, 119, 38, 162, 169]
v_syst_list = [3.85]# [4.75, 5.7, 5, 4, 3.85]#, -, -, -, -, -]
distance_list = [123]# [121, 101, 134, 161, 123, 138, 116, 155, 148, 165]

for i in range(1):#len(source_list)):
    '''
    # for multiple sources
    continuum = casa.Cube(str(DSHARP_list[i])+'_continuum.fits')
    source = casa.Cube(str(DSHARP_list[i])+'_CO.fits')

    nx = source.nx
    cont = np.nan_to_num(continuum.image[0,:,:])
    cont = resize(cont, (nx,nx))
    cont = np.array(rotate(cont, PA_list[i] - 90.0, reshape=False))
    star_coordinates = np.where(cont == np.amax(cont))
    listofcordinates = list(zip(star_coordinates[0], star_coordinates[1]))
    print(DSHARP_list[i])
    for cord in listofcordinates:
        print('coordinates of maximum in continuum image (y,x) = '+str(cord))

        if plot_continuum is True:
            plt.imshow(cont, origin='lower')
            plt.plot(cord[1],cord[0], '.', color='red')
            plt.savefig(str(DSHARP_list[i])+'_continuum.png')
    '''
    source = casa.Cube(str(DSHARP_list[i])+'_CO.fits')
    y_star = 720#cord[0] 
    x_star = 719#cord[1] 
    inc = inc_list[i]
    PA = PA_list[i]
    v_syst = v_syst_list[i] 
    distance = distance_list[i]
    sigma = 5

    list1 = ['height', 'velocity', 'brightness temperature']
    list2 = ['r (au)', 'h (au)', 'v (km/s)', 'Tb (K)']
    
    for k in range(3):
        plt.figure(list1[k])
        plt.clf()
        plt.xlabel(list2[0], labelpad=7)
        plt.ylabel(list2[k+1], labelpad=7)
    
    #y_star, x_star = measure_height.star_location(continuum, source, PA_list, i)

    n, x, y, T = measure_height.detect_surface(source, PA=PA, plot=plot_rotated_cube, sigma=sigma, y_star=y_star)

    r, h, v, Tb = measure_height.measure_mol_surface(source, n, x, y, T, inc=inc, x_star=x_star, y_star=y_star, v_syst=v_syst, distance=distance)
    
    measure_height.plotting_mol_surface(r, h, v, Tb, i, isotope)
    
    if plot_layers is True:
        nv = source.nv
        for j in range(nv):
            iv = j 
            measure_height.plot_surface(source, n, x, y, T, iv, PA=PA, win=20)
            plt.plot(x_star,y_star, '.', color='white')
            plt.savefig('DSHARP_Layers/'+str(DSHARP_list[i])+'_channel_'+str(j)+'.png')
            
    for k in range(3):
        plt.figure(list1[k])
        plt.legend(loc='best')
        plt.savefig(str(DSHARP_list[i])+'_'+str(list1[k])+'.png')
    
    plt.show()
    
