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


def detect_surface(cube, PA=None, plot=False, sigma=None, y_star=None, x_star=None):
    
    nx, nv = cube.nx, cube.nv
    
    n_surf = np.zeros(nv, dtype=int)
    x_surf = np.zeros([nv,nx])
    y_surf = np.zeros([nv,nx,2])
    Tb_surf = np.zeros([nv,nx,2])
    P = np.zeros([nv,2])
    x_old = np.zeros([nv,nx])
    y_old = np.zeros([nv,nx,2])
    n0_surf = np.zeros(nv, dtype=int)
    
    ### measure rms in the 1st channel 
    std = np.nanstd(cube.image[1,:,:])

    ### stepping through each channel map
    for iv in range(nv):

        if iv == nv-1:
            print(f'total number of channels = {nv}')
            
        #print('channel '+str(iv+1)+" of "+str(nv))

        ### rotate the image so semi-major axis is aligned with x-axis
        im = np.nan_to_num(cube.image[iv,:,:])
        if PA is not None:
            im = np.array(rotate(im, PA - 90.0, reshape=False))

        ### plotting rotated channel map (if needed)
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
        
        ### looping through each x-coordinate
        for i in range(nx):
            vert_profile = im[:,i]

            ### finding the flux maxima for each slice in the x-axis
            j_max = search_maxima(vert_profile, y_star, threshold=sigma*std, dx=cube.bmaj/cube.pixelscale)
                
            ### require a minimum of 2 points; to identify surfaces above and below the star 
            if (len(j_max) > 1) & (abs(i - x_star) < 300): 
                in_surface[i] = True
                
                ### storing only the 2 brightest maxima. want j[i,0] to be below the star and j[i,1] to be above.
                j_surf[i,:] = j_max[:2]
                
                ### in case both the brightest points are on one side of the disk
                 
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

                if ((j_surf[i,1] - y_star) > 300) or ((y_star - j_surf[i,0]) > 300):
                    in_surface[i] = False
                
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

        # cleaning each channel map
        if np.any(in_surface):
            
            x = np.arange(nx) 

            n0 = np.sum(in_surface)
            n0_surf[iv] = n0
            if n0 > 0:
                x_old[iv,:n0] = x[in_surface]
                y_old[iv,:n0,:] = j_surf_exact[in_surface,:]
            
            x1 = x[in_surface]
            y1 = np.mean(j_surf_exact[in_surface,:],axis=1)
            y0 = j_surf_exact[in_surface,:]
            
            if np.size(x1) > 10:
                limit = int(np.size(x1)/4)
            else:
                limit = int(np.size(x1))

            if np.size(x1) > 10:
                if (np.mean(x1) < x_star):
                    limit = int(3*limit)
                    x2 = x1[limit:]
                    y2 = y1[limit:]
                else:
                    x2 = x1[:limit]
                    y2 = y1[:limit]
            else:
                x2 = x1[:limit]
                y2 = y1[:limit]
            
            P[iv,:] = np.polyfit(x2,y2,1)

            trend_limit = abs(np.mean(j_surf_exact[:,:],axis=1) - (P[iv,0]*x + P[iv,1])) / (y0[limit-1,1] - y0[limit-1,0])
            
            star_limit = abs(y_star - (P[iv,0]*x_star + P[iv,1])) / (y0[limit-1,1] - y0[limit-1,0]) #(P[iv,0]*x_star + P[iv,1])
            
            in_surface = in_surface & (trend_limit < 0.10) & (star_limit < 0.10)
            
            # Saving the data
            n = np.sum(in_surface)
            n_surf[iv] = n # number of points in that surface
            if n > 0:
                x_surf[iv,:n] = x[in_surface]
                y_surf[iv,:n,:] = j_surf_exact[in_surface,:]
                Tb_surf[iv,:n,:] = T_surf[in_surface,:]
                

    return n_surf, x_surf, y_surf, Tb_surf, P, x_old, y_old, n0_surf



def plot_surface(cube, n, x, y, Tb, iv, P, x_old, y_old, n0, PA=None, win=20):

    im = np.nan_to_num(cube.image[iv,:,:])
    if PA is not None:
        im = np.array(rotate(im, PA - 90.0, reshape=False))

    plt.figure(win)
    plt.clf()
    plt.imshow(im, origin="lower")#, interpolation="bilinear")  

    y_mean = np.mean(y[:,:,:],axis=2)

    y0_mean = np.mean(y_old[:,:,:],axis=2)

    if n0[iv]:
        plt.plot(x_old[iv,:n0[iv]],y_old[iv,:n0[iv],0],"o",color="fuchsia",markersize=1.5, label='B.S. points removed')
        plt.plot(x_old[iv,:n0[iv]],y_old[iv,:n0[iv],1],"o",color="cyan",markersize=1.5, label='T.S. points removed')
        plt.plot(x_old[iv,:n0[iv]],y0_mean[iv,:n0[iv]],"o",color="yellow",markersize=1.5, label='avg. of points removed')
        plt.plot(x_old[iv,:n0[iv]], P[iv,0]*x_old[iv,:n0[iv]] + P[iv,1], '--', color='black', markersize=0.5, label='trend line')
    
    if n[iv]:
        plt.plot(x[iv,:n[iv]],y[iv,:n[iv],0],"o",color="red",markersize=1.5, label='B.S. points kept')
        plt.plot(x[iv,:n[iv]],y[iv,:n[iv],1],"o",color="blue",markersize=1.5, label='T.S. points kept')
        plt.plot(x[iv,:n[iv]],y_mean[iv,:n[iv]],"o",color="orange",markersize=1.5, label='avg. of points kept')

    if n0[iv]:
        # zoom-in on the detected surfaces
        plt.xlim(np.min(x_old[iv,:n0[iv]]) - 5*cube.bmaj/cube.pixelscale, np.max(x_old[iv,:n0[iv]]) + 5*cube.bmaj/cube.pixelscale)
        plt.ylim(np.min(y_old[iv,:n0[iv],:]) - 5*cube.bmaj/cube.pixelscale, np.max(y_old[iv,:n0[iv],:]) + 5*cube.bmaj/cube.pixelscale)
        #adding a legend
        plt.legend(loc='best', prop={'size': 6})
    
        

def search_maxima(y, y_star, threshold=None, dx=0):
    ### passing im[:] as y[:] here ###
    '''
    i_max = []

    for k in range(2):
        #print(k)
        if k == 0:
            yp = y[:y_star] 
        elif k == 1:
            yp = y[y_star:] + y_star
        #print(y)
        dy = yp[1:] - yp[:-1]

        y_max = np.where((np.hstack((0, dy)) > 0) & (np.hstack((dy, 0)) < 0))[0]

        if threshold is not None:
            y_max = y_max[np.where(yp[y_max]>threshold)]

        y_max = y_max[np.argsort(yp[y_max])][::-1] 

        if np.size(y_max) > 0:
            if dx > 1:
                flag_remove = np.zeros(np.size(y_max), dtype=bool)
                for i in range(np.size(y_max)):
                    if not flag_remove[i]:
                        flag_remove = flag_remove | (y_max >= y_max[i] - dx) & (y_max <= y_max[i] + dx)
                        flag_remove[i] = False # Keep current max
                y_max = y_max[~flag_remove]

            #print(y_max)
            i_max.append(y_max[0])
            
        print(i_max)

    
    y_max_a = np.where(y[y_star:]>threshold)[0] + y_star

    ### sorting y-coordinates from highest to lowest in flux
    
    if np.size(y_max_a) > 0:
        if dx > 1:
            flag_remove = np.zeros(np.size(y_max_a), dtype=bool)
            for i in range(np.size(y_max_a)):
                if not flag_remove[i]:
                    flag_remove = flag_remove | (y_max_a >= y_max_a[i] - dx) & (y_max_a <= y_max_a[i] + dx)
                    flag_remove[i] = False # Keep current max
            y_max_a = y_max_a[~flag_remove]
    
    y_max_b = np.where(y[:y_star]>threshold)[0]

    if np.size(y_max_b) > 0:
        if dx > 1:
            flag_remove = np.zeros(np.size(y_max_b), dtype=bool)
            for i in range(np.size(y_max_b)):
                if not flag_remove[i]:
                    flag_remove = flag_remove | (y_max_b >= y_max_b[i] - dx) & (y_max_b <= y_max_b[i] + dx)
                    flag_remove[i] = False # Keep current max
            y_max_b = y_max_b[~flag_remove]

    if ((len(y_max_a)>1) & (len(y_max_b)>1)):
        i_max = []
        i_max.append((y_max_b[np.argsort(y[y_max_b])][::-1])[0])
        i_max.append((y_max_a[np.argsort(y[y_max_a])][::-1])[0])
        
    else:
        i_max = []

    #print(i_max)   
    '''
    
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
    
