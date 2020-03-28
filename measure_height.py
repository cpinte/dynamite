import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, shift
from scipy.interpolate import interp1d
import scipy.constants as sc
from scipy.stats import binned_statistic
from astropy.io import fits
from skimage.transform import resize

##############################################################################

def measure_mol_surface(cube, n, x, y, T, inc=None, x_star=None, y_star=None, v_syst= None, distance=None, optimize_x_star=None):

    ### some things to note:
        # value of x_star plays on dispersion of h and v
        # value of PA creates a huge dispersion
        # value of y_star shift the curve h(r) vertically, massive change on flaring exponent
        # value of inc changes the shape a bit, but increases dispersion on velocity
        # a = above , b = below

    np.set_printoptions(threshold=np.inf) 

    ### computing the radius and height 
    y_a = y[:,:,1]                        # y[channel number, x value, y value above star]
    y_b = y[:,:,0]                        # y[channel number, x value, y value below star]

    y_c = (y_a + y_b) / 2.
    mask = (y_c == 0)
    y_centre = np.ma.masked_array(y_c,mask).compressed()
    
    if (len(np.where(y_centre.ravel()<y_star)[0]) > 0.5*len(y_centre.ravel())):
        print('upper layer below y_star')
        h = -(y_c - y_star) / np.sin(inc)
        r = np.sqrt((x - x_star)**2 + ((y_c - y_b)/np.cos(inc))**2)
    else:
        print('upper layer above y_star')
        h = (y_c - y_star) / np.sin(inc)
        r = np.sqrt((x - x_star)**2 + ((y_a - y_c)/np.cos(inc))**2)
    
    v = (cube.velocity[:,np.newaxis] - v_syst) * r / ((x - x_star) * np.sin(inc))

    B = np.mean(T[:,:,:],axis=2)
    
    r *= cube.pixelscale
    h *= cube.pixelscale
    if distance is not None:
        r *= distance
        h *= distance

    mask1 = np.isinf(v) | np.isnan(v)
    print(f'no. of channels with invalid velocities removed = {np.sum(mask1)}')
    mask2 = mask1 | (np.abs(cube.velocity[:,np.newaxis] - v_syst) < 0.4)
    print(f'no. of points outside filter removed = {np.sum(mask2)-np.sum(mask1)}')
    mask3 = mask2 | (h<0) | (r>400)

    r = np.ma.masked_array(r,mask3).compressed()
    h = np.ma.masked_array(h,mask3).compressed()
    v = np.ma.masked_array(v,mask3).compressed()
    B = np.ma.masked_array(B,mask3).compressed()

    if (np.mean(v) < 0):
        v = -v
    
    mask4 = (v<0)
    print(f'no. of outliers removed = {(np.sum(mask3)-np.sum(mask2))+np.sum(mask4)}')
    
    r = np.ma.masked_array(r,mask4).compressed()
    h = np.ma.masked_array(h,mask4).compressed()
    v = np.ma.masked_array(v,mask4).compressed()
    B = np.ma.masked_array(B,mask4).compressed()
    
    Tb = cube._Jybeam_to_Tb(B)

    print(f'no. of plots left for plotting = {len(r.ravel())}')
    
    return r, h, v, Tb



def plotting_mol_surface(r, h, v, Tb, i, isotope):

    bins = 70
    plot = ['height', 'velocity', 'brightness temperature']
    var = [h, v, Tb]
    stat = ['mean', 'mean', 'max']

    for k in range(3):
        plt.figure(plot[k])

        data,_,_ = binned_statistic(r, [r, var[k]], statistic=stat[k], bins=bins)
        std,_,_ = binned_statistic(r, var[k], statistic='std', bins=bins)

        plt.scatter(data[0,:], data[1,:], alpha=0.7, s=5, label=isotope[i])
        plt.errorbar(data[0,:], data[1,:], yerr=std, ls='none')
    
    ### polynomial fitting
    '''
    plt.figure('height') ###
    
    P, res_h, _, _, _ = np.ma.polyfit(np.log10(r.ravel()),np.log10(h.ravel()), 1, full=True)
    #x = np.linspace(np.min(r), 300, 100)
    x = np.linspace(np.min(r),np.max(r),100)
    
    plt.plot(x, 10**P[1] * x**P[0])
    '''
    

    
class Surface(dict):
    """ Represents the set of points defining the molecular surface
    extracted from a cube

    n : int
        Number of abscices where data points were extracted

    x : int
        Abcises of the points

    y : ndarray(float,2)
        Ordinates of the points

    T : ndarray(float,2)
        Brigtness temperature of the points

    PA : best PA found

    Notes
    -----
    There may be additional attributes not listed above depending of the
    specific solver. Since this class is essentially a subclass of dict
    with attribute accessors, one can see which attributes are available
    using the `keys()` method.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


def detect_surface(cube, PA=None, plot=False, sigma=None, y_star=None, win=20):
    
    nx, nv = cube.nx, cube.nv
    
    n_surf = np.zeros(nv, dtype=int)
    x_surf = np.zeros([nv,nx])
    y_surf = np.zeros([nv,nx,2])
    Tb_surf = np.zeros([nv,nx,2])

    surface_color = ["red","blue"]
    
    ### measure rms in the 1st channel 
    std = np.nanstd(cube.image[1,:,:])

    ### stepping through each channel map
    for iv in range(nv):

        if iv == nv-1:
            print('total number of channels = '+str(nv))
            
        #print('channel '+str(iv+1)+" of "+str(nv))

        ### rotate the image so semi-major axis is aligned with x-axis
        im = np.nan_to_num(cube.image[iv,:,:])
        if PA is not None:
            im = np.array(rotate(im, PA - 90.0, reshape=False))

        ### plotting rotated channel map
        if plot is True:
            if iv==0:
                img_rotated_list=[]

            img_rotated_list.append(im)
            img_rotated_array = np.array(img_rotated_list)

            if iv==nv-1:
                fits.writeto('CO_rotated.fits', img_rotated_array, overwrite=True)
        
        ### setting up arrays in each channel map
        in_surface = np.full(nx,False)         #creates an array the size of nx without a fill value (False).
        j_surf = np.zeros([nx,2], dtype=int)
        j_surf_exact = np.zeros([nx,2])
        T_surf = np.zeros([nx,2])
        
        ### looping through each x-coordinate in each channel map
        for i in range(nx):
            vert_profile = im[:,i]

            ### finding the flux maxima for each slice in the x-axis
            j_max = search_maxima(vert_profile,threshold=sigma*std, dx=cube.bmaj/cube.pixelscale)
                
            ### require a minimum of 2 points; to identify surfaces above and below the star 
            if len(j_max) > 1: 
                in_surface[i] = True
                
                ### storing only the 2 brightest maxima.
                ### want j[i,0] to be below the star and j[i,1] to be above.
                j_surf[i,:] = j_max[:2]
                
                ### in case both the brightest points are on one side of the disk
                #if y_star is not None:
                    
                if (j_surf[i,0] < y_star and j_surf[i,1] < y_star):
                    j_max_2nd = j_max[np.where(j_max > y_star)]
                    if len(j_max_2nd) > 0:
                        j_surf[i,1] = j_max_2nd[0]
                    else:
                        in_surface[i] = False
                            
                elif (j_surf[i,0] > y_star and j_surf[i,1] > y_star):
                    j_max_2nd = j_max[np.where(j_max < y_star)]
                    if len(j_max_2nd) > 0:
                        j_surf[i,0] = j_max_2nd[0]
                    else:
                        in_surface[i] = False
                else:
                    j_surf[i,:] = np.sort(j_max[:2])
                  
                ### refining position of maxima using a spatial quadratic
                for k in range(2):
                    j = j_surf[i,k]

                    f_max = im[j,i]
                    f_minus = im[j-1,i]
                    f_plus = im[j+1,i]

                    # Work out the polynomial coefficients
                    a0 = 13. * f_max / 12. - (f_plus + f_minus) / 24.
                    a1 = 0.5 * (f_plus - f_minus)
                    a2 = 0.5 * (f_plus + f_minus - 2*f_max)

                    # Compute the maximum of the quadratic
                    y_max = j - 0.5 * a1 / a2
                    f_max = a0 - 0.25 * a1**2 / a2

                    # Saving the coordinates
                    j_surf_exact[i,k] = y_max
                    T_surf[i,k] = f_max 
       
        #-- We test if front side is too high or the back side too low
        # this happens when the data gets noisy or diffuse and there are local maxima
        # fit a line to average curve and remove points from front if above average
        # and from back surface if below average (most of these should have been dealt with test on star position)

        if np.any(in_surface):
            x = np.arange(nx)
            x1 = x[in_surface]

            y1 = np.mean(j_surf_exact[in_surface,:],axis=1)
            P = np.polyfit(x1,y1,1)

            #x_plot = np.array([0,nx])
            #plt.plot(x_plot, P[1] + P[0]*x_plot)

            in_surface_tmp = in_surface & (j_surf_exact[:,0] < (P[1] + P[0]*x)) # test only front surface
            in_surface_tmp = in_surface & (j_surf_exact[:,0] < (P[1] + P[0]*x)) & (j_surf_exact[:,1] > (P[1] + P[0]*x))

            # We remove the weird point and reddo the fit again to ensure the slope we use is not too bad
            x1 = x[in_surface_tmp]
            y1 = np.mean(j_surf_exact[in_surface_tmp,:],axis=1)
            P = np.polyfit(x1,y1,1)

            #in_surface = in_surface &  (j_surf_exact[:,0] < (P[1] + P[0]*x)) # test only front surface
            in_surface = in_surface & (j_surf_exact[:,0] < (P[1] + P[0]*x)) & (j_surf_exact[:,1] > (P[1] + P[0]*x))

            # Saving the data
            n = np.sum(in_surface)
            n_surf[iv] = n # number of points in that surface
            if n > 0:
                x_surf[iv,:n] = x[in_surface]
                y_surf[iv,:n,:] = j_surf_exact[in_surface,:]
                Tb_surf[iv,:n,:] = T_surf[in_surface,:]

    
    return n_surf, x_surf, y_surf, Tb_surf



def plot_surface(cube, n, x, y, Tb, iv, PA=None, win=20):

    im = np.nan_to_num(cube.image[iv,:,:])
    if PA is not None:
        im = np.array(rotate(im, PA - 90.0, reshape=False))

    plt.figure(win)
    plt.clf()
    plt.imshow(im, origin="lower")#, interpolation="bilinear")

    y_mean = np.mean(y[:,:,:],axis=2)
    
    if n[iv]:
        plt.plot(x[iv,:n[iv]],y[iv,:n[iv],0],"o",color="red",markersize=1)
        plt.plot(x[iv,:n[iv]],y[iv,:n[iv],1],"o",color="blue",markersize=1)
        plt.plot(x[iv,:n[iv]],y_mean[iv,:n[iv]],"o",color="yellow",markersize=1)

        # We zoom on the detected surfaces
        plt.xlim(np.min(x[iv,:n[iv]]) - 10*cube.bmaj/cube.pixelscale,np.max(x[iv,:n[iv]]) + 10*cube.bmaj/cube.pixelscale)
        plt.ylim(np.min(y[iv,:n[iv],:]) - 10*cube.bmaj/cube.pixelscale,np.max(y[iv,:n[iv],:]) + 10*cube.bmaj/cube.pixelscale)



def search_maxima(y, threshold=None, dx=0):
    ### passing im[:] as y[:] here ###

    ### measuring the change in flux between y[i] and y[i-1]
    dy = y[1:] - y[:-1]

    ### finding maxima. a positive dy followed by a negative dy. stores all the points where this happens, don't worry about notation. 
    i_max = np.where((np.hstack((0, dy)) > 0) & (np.hstack((dy, 0)) < 0))[0]
    
    ### filtering out only y-coordinates above a signal to noise threshold
    if threshold:
        i_max = i_max[np.where(y[i_max]>threshold)]

    ### sorting y-coordinates from highest to lowest in flux
    i_max = i_max[np.argsort(y[i_max])][::-1]

    ### signals must be separated by at least a beam size. to ensure signal is resolved
    if i_max.size > 0:
        if dx > 1:
            flag_remove = np.zeros(i_max.size, dtype=bool)
            for i in range(i_max.size):
                if not flag_remove[i]:
                    flag_remove = flag_remove | (i_max >= i_max[i] - dx) & (i_max <= i_max[i] + dx)
                    flag_remove[i] = False # Keep current max
                    # remove the unwanted maxima
            i_max = i_max[~flag_remove]

    return i_max



def star_location(source, continuum, PA=False, plot=False, name=False):

    nx = source.nx
    
    continuum = np.nan_to_num(continuum.image[0,:,:])
    continuum = resize(continuum, (nx,nx))
    continuum = np.array(rotate(continuum, PA - 90.0, reshape=False))
    star_coordinates = np.where(continuum == np.amax(continuum))
    listofcordinates = list(zip(star_coordinates[0], star_coordinates[1]))
   
    for cord in listofcordinates:
        print('coordinates of maximum in continuum image (y,x) = '+str(cord))

    y_star = cord[0] 
    x_star = cord[1]
    
    if plot is True:
        plt.figure('continuum')
        plt.clf()
        plt.imshow(continuum, origin='lower')
        plt.plot(cord[1],cord[0], '.', color='red')
        plt.savefig(f'{name}_continuum.png')

    return y_star, x_star
    
