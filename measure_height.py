from scipy.ndimage import rotate, shift
from scipy.interpolate import interp1d
import scipy.constants as sc
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import binned_statistic

import matplotlib.pyplot as plt
import bettermoments as bm
import my_casa_cube as casa

class Surface:

    def __init__(self, cube=None, PA=None, inc=None, x_c=None, y_c=None, v_syst=None, sigma=5, **kwargs):

        self.cube = cube

        self.PA = PA
        self.inc = inc
        self.x_c = x_c
        self.y_c = y_c
        self.sigma = sigma
        self.v_syst = v_syst

        if not os.path.exists(str(cube.filename.replace(".fits","_gv0.fits"))):
            self._compute_velocity_fields()
        else:
            path = self.filename
            data_bm, velax_bm = bm.load_cube(path)
            rms = bm.estimate_RMS(data=data_bm, N=1)
        
        self._detect_surface()
        self._compute_surface()
        self._plot_mol_surface()
        self._plot_traced_channels()

        return
    

    def _compute_velocity_fields(self):
        """
        For computing the line of sight velocity fields using the bettermoments package
        """

        path = self.cube.filename
        data_bm, velax_bm = bm.load_cube(path)

        rms = bm.estimate_RMS(data=data_bm, N=1)
        
        moments = bm.collapse_gaussian(velax=bm_velax, data=bm_data, rms=rms)

        bm.save_to_FITS(moments=moments, method='gaussian', path=path)

        self.rms = rms
        
        
    def _detect_surface(self):
        """
        Infer the upper emission surface from the provided cube
        extract the emission surface in each channel and loop over channels

        Args:
        cube (casa instance): An imgcube instance of the line data.

        PA (float): Position angle of the source in [degrees].
        y_star (optional) : position of star in  pixel (in rotated image), used to filter some bad data
        without y_star, more points might be rejected
        """

        # line cube
        cube = self.cube

        # parameters
        nx, nv = cube.nx, cube.nv
        dv = round(cube.velocity[1]-cube.velocity[0], 2)

        # setting up arrays
        x_surf = np.zeros([nv,nx])
        y_surf = np.zeros([nv,nx,2])
        Tb_surf = np.zeros([nv,nx,2])

        # load in velocity map
        mom9 = casa.Cube(str(cube.filename.replace(".fits","_gv0.fits")))
        mom9_im = np.nan_to_num(mom9.image[:,:])
        mom9_im_rot = rotate_disc(mom9_im, PA=self.PA, x_c=self.x_c, y_c=self.y_c) 

        # Loop over the channels
        for iv in range(nv):
            
            print(iv,"/",nv-1, v[iv][0])
            
            # rotate the image so major axis is aligned with x-axis.
            im = np.nan_to_num(cube.image[iv,:,:])
            im_rot = rotate_disc(im, PA=self.PA, x_c=self.x_c, y_c=self.y_c)

            # setting up arrays in each channel
            in_surface = np.full(nx, False)
            local_surf = np.zeros([nx,2], dtype=int)
            local_exact = np.zeros([nx,2])
            B_surf = np.zeros([nx,2])

            # loop through each x-coordinate
            for i in range(nx):

                vert_profile = im_rot[:,i]
                mom9_profile = mom9_im_rot[:,i]
            
                # finding the flux maxima for each slice in the x-axis
                local_max, mom9_coords = search_maxima(vert_profile, mom9_profile, v=v[iv][0], dv=dv, y_c=self.y_c, threshold=self.sigma*self.rms, dx=cube.bmaj/cube.pixelscale)

                # require a minimum of 2 points; to identify surfaces above and below the star
                if len(local_max) > 1:
                    in_surface[i] = True
                                
                    # defining the two surfaces. local_surf[i,0] is below y_star, and local_surf[i,1] is above y_star.
                    local_surf[i,:] = local_max[:2]

                    for k in range(2):
                        j = local_surf[i,k]
                        B_surf[i,k] = im[j,i]

            # Saving the data
            if np.any(in_surface):
            
                x = np.arange(nx)            
                n = np.sum(in_surface)
                if n > 0:
                    x_surf[iv,:n] = x[in_surface]
                    y_surf[iv,:n,:] = local_surf[in_surface,:]
                    Bv_surf[iv,:n,:] = B_surf[in_surface,:]
    

        self.x_surf = x_surf
        self.y_surf = y_surf
        self.Bv_surf = Bv_surf


    def _compute_surface(self):
        """
        Deproject the detected surfaces to estimate, r, h, and v of the emitting layer

        inc (float): Inclination of the source in [degrees].
        """

        ### some things to note:
        # Impact of parameters :
        # - value of x_star plays on dispersion of h and v
        # - value of PA creates a huge dispersion
        # - value of y_star shift the curve h(r) vertically, massive change on flaring exponent
        # - value of inc changes the shape a bit, but increases dispersion on velocity

        inc_rad = np.radians(self.inc)

        # y-coordinates for surfaces vertically above and below disc centre in sky coordinates
        y_a = y_surf[:,:,1] - y_c
        y_b = y_surf[:,:,0] - y_c

        # determining which surface (top/bottom) is the near/far side.
        y_mean = np.mean(y_surf[:,:,:], axis=2) - y_c
        mask = (y_mean == 0)    # removing x coordinates with no traced points.
        y_mean_masked = np.ma.masked_array(y_mean, mask).compressed()
        
        if (len(np.where(y_mean_masked.ravel() < y_c)[0]) > 0.5*len(y_mean_masked.ravel())):
            factor = 0
        else:
            factor = 1

        # computing the radius and height and converting units to arcseconds
        if factor == 0:
            print('bottom layer is the near side')
            h = y_mean / np.sin(inc_rad)
            r = np.hypot(self.x_surf - self.x_c, abs(y_b - y_mean) / np.cos(inc_rad))
        elif factor == 1:
            print('top layer is the near side')
            h = y_mean / np.sin(inc_rad)
            r = np.hypot(self.x_surf - self.x_c, (y_a - y_mean) / np.cos(inc_rad))

        r *= self.cube.pixelscale
        h *= self.cube.pixelscale        

        # computing velocity and brightness profile
        v = (self.cube.velocity[:,np.newaxis] - self.v_syst) * r / ((self.x_surf - self.x_c) * np.sin(inc_rad))
        Bv = np.mean(self.Bv_surf[:,:,:], axis=2)
        # check is the disc is rotating in the opposite direction
        if (np.mean(v) < 0):
            v *= -1

        # masking invalid points
        mask1 = np.isinf(v) | np.isnan(v) | (h<0) | (v<0)

        r = np.ma.masked_array(r,mask1).compressed()
        h = np.ma.masked_array(h,mask1).compressed()
        v = np.ma.masked_array(v,mask1).compressed()
        Bv = np.ma.masked_array(Bv,mask1).compressed()

        # compute brightness temperature
        Tb = self.cube._Jybeam_to_Tb(Bv)

        self.r = r
        self.h = h
        self.v = v
        self.Tb = Tb


    def _plot_mol_surface(self):
        bins = 70
        plot = ['h', 'v', 'Tb']
        units = ['[arcsec]', '[km/s]', '[K]']
        var = [h, v, Tb]
        stat = ['mean', 'mean', 'max']
        ax = [ax1, ax2, ax3]

        freq = str(round(self.cube.restfreq/1.e9))
        source = self.cube.object
        location = os.path.dirname(os.path.realpath(self.cube.filename))
        
        for k in range(3):
            
            data,_,_ = binned_statistic(r, [r, var[k]], statistic=stat[k], bins=bins)
            std,_,_ = binned_statistic(r, var[k], statistic='std', bins=bins)

            ax[k].scatter(data[0,:], data[1,:], alpha=0.7, s=5, label=isotope)
            ax[k].errorbar(data[0,:], data[1,:], yerr=std, ls='none')

            ax[k].set_xlabel('r [arcsec]')
            ax[k].set_ylabel(plot[k]+units[k])

            np.savetxt(location+'/'+source+'_'+freq+'_surface_params.txt', np.column_stack([r, h, v, Tb]))

            plt.savefig(location+'/'+source+'_'+freq+'_plot[k]_vs_r.pdf', bbox_inches='tight')

            plt.close()
            

    def _plot_traced_channels(self):

        # tidy-up arrays
        self.x_surf[self.x_surf==0] = np.nan
        self.y_surf[self.y_surf==0] = np.nan
            
        # converting traced points from pixels to arcseconds
        x_arc = -(self.x_surf - ((self.cube.nx - 1) / 2)) * self.cube.pixelscale
        y_arc = (self.y_surf - ((self.cube.ny - 1) / 2)) * self.cube.pixelscale

        # converting star location from pixels to arcseconds
        xc_arc = -(self.x_c - ((self.cube.nx - 1) / 2)) * self.cube.pixelscale
        yc_arc = (self.y_c - ((self.cube.ny - 1) / 2)) * self.cube.pixelscale
        
        ############
        nv = self.cube.nv

        norm = PowerNorm(1, vmin=0, vmax=np.max(self.cube._Jybeam_to_Tb(np.nan_to_num(data.image[:,:,:]))))
        cmap = cmo.cm.rain

        freq = str(round(self.cube.restfreq/1.e9))
        source = self.cube.object
        location = os.path.dirname(os.path.realpath(self.cube.filename))
        if not os.path.exists(location+'/'+source+'_layers/'):
            os.mkdir(location+'/'+source+'_layers/')
        
        for iv in range(nv):

            fig = plt.figure(figsize=(6,6))
            gs = gridspec.GridSpec(1,1)
            ax = plt.subplot(gs[0])
                     
            im_K = self.cube._Jybeam_to_Tb(np.nan_to_num(self.cube.image[iv,:,:]))

            im = rotate_disc(im_K, PA=self.PA, x_c=self.x_c, y_c=self.y_c) 

            image = ax.imshow(im, origin='lower', cmap=cmap, norm=norm, extent=self.cube.extent - np.asarray([xc_arc,xc_arc,yc_arc,yc_arc]))

            # adding marker for disc centre
            ax.plot(xc_arc, yc_arc, '+', color='white')

            ## adding trace points                
            ax.plot(x_arc[iv,:],y_arc[iv,:,0], '.', markersize=2, color='white')
            ax.plot(x_arc[iv,:],y_arc[iv,:,1], '.', markersize=2, color='white')

            # zooming in on the surface
            plt.xlim(np.min(x_arc[iv]) - 0.2*(abs(np.min(x_arc[iv]) - self.x_c)), np.max(x_arc[iv]) + 0.2*(abs(np.max(x_arc[iv]) - self.x_c)))
            plt.ylim(np.min(y_arc[iv]) - 0.2*(abs(np.min(y_arc[iv]) - self.y_c)), np.max(y_arc[iv]) + 0.2*(abs(np.max(y_arc[iv]) - self.y_c)))
    
            # adding beam
            ax = plt.gca()
            beam = Ellipse(xy=(np.max(x_arc[iv]) + 0.17*(abs(np.max(x_arc[iv]) - self.x_c)),np.min(y_arc[iv]) - 0.17*(abs(np.min(y_arc[iv]) - self.y_c))), width=self.cube.bmin, height=self.cube.bmaj, angle=-self.cube.bpa, fill=True, color='white')
            ax.add_patch(beam)

            plt.savefig(location+'/'+source+'_layers/'+source+'_'+freq+'_channel_'+str(iv)+'.pdf', bbox_inches='tight')

            plt.close()


def search_maxima(yprofile, velprofile, v=None, dv=dv, y_c=None, threshold=None, dx=0):

    # find local maxima
    v += dv/2
    
    dy = yprofile[1:] - yprofile[:-1]
    i_max = np.where((np.hstack((0, dy)) > 0) & (np.hstack((dy, 0)) < 0))[0]
    
    if threshold:
        # to remove noise
        i_max = i_max[np.where(yprofile[i_max]>threshold)]
        # to remove loud noise
        i_max = i_max[np.where(yprofile[i_max-1]>threshold)]
        i_max = i_max[np.where(yprofile[i_max+1]>threshold)]

    # re-arrange local maxima from strongest signal to weakest
    i_max = i_max[np.argsort(yprofile[i_max])][::-1]
    
    # finding maxima above and below y_star
    i_max_above = i_max[np.where(i_max > y_c)]
    i_max_below = i_max[np.where(i_max < y_c)]
    
    # finding isovelocity points, from velocity fields map
    diff_vel = abs(velprofile - v)
    diff_vel[diff_vel > dv/2] = np.inf

    vel_above = np.argmin(diff_vel[y_c:]) + y_c
    vel_below = np.argmin(diff_vel[:y_c])
        
    vel_coords = [vel_below, vel_above] 
    
    # finding point closest to isovelocity curve
    if len(i_max_above) > 0 and len(i_max_below) > 0:
        ycoord_above = i_max_above[np.where(abs(i_max_above - vel_above) < dx)]
        ycoord_below = i_max_below[np.where(abs(i_max_below - vel_below) < dx)]
    else:
        ycoord_above = []
        ycoord_below = []

    # defining the i_max array to return
    if len(ycoord_above) > 0 and len(ycoord_below) > 0:
        ycoords = [ycoord_below[0], ycoord_above[0]]
    else:
        ycoords = []
    
    return ycoords, vel_coords


'''
def search_maxima(y, threshold=None, dx=0):
    """
    Returns the indices of the maxima of a function
    Indices are sorted by decreasing values of the maxima

    Args:
         y : array where to search for maxima
         threshold : minimum value of y for a maximum to be detected
         dx : minimum spacing between maxima [in pixel]
    """

    # A maxima is a positive dy followed by a negative dy
    dy = y[1:] - y[:-1]
    i_max = np.where((np.hstack((0, dy)) > 0) & (np.hstack((dy, 0)) < 0))[0]

    if threshold:
        i_max = i_max[np.where(y[i_max]>threshold)]

    # Sort the peaks by height
    i_max = i_max[np.argsort(y[i_max])][::-1]

    # detect small maxima closer than minimum distance dx from a higher maximum
    if i_max.size:
        if dx > 1:
            flag_remove = np.zeros(i_max.size, dtype=bool)
            for i in range(i_max.size):
                if not flag_remove[i]:
                    flag_remove = flag_remove | (i_max >= i_max[i] - dx) & (i_max <= i_max[i] + dx)
                    flag_remove[i] = False # Keep current max
                    # remove the unwanted maxima
            i_max = i_max[~flag_remove]

    return i_max
'''

def rotate_disc(channel, PA=None, x_c=None, y_c=None):

    """
    For rotating map around defined disc centre
    """
    
    if PA is not None:
        padX = [channel.shape[1] - x_c, x_c]
        padY = [channel.shape[0] - y_c, y_c]
        imgP = np.pad(channel, [padY, padX], 'constant')
        imgR = ndimage.rotate(imgP, PA - 90, reshape=False)
        im = imgR[padY[0] : -padY[1], padX[0] : -padX[1]]
    else:
        print('Error! Need to specify a PA')
        sys.exit() 

    return im  
