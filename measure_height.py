from scipy import ndimage
from scipy.ndimage import rotate
from scipy.interpolate import interp1d
import scipy.constants as sc
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import binned_statistic
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from matplotlib.colors import LogNorm, PowerNorm

import matplotlib.pyplot as plt
import bettermoments as bm
import my_casa_cube as casa
import os
import sys
import cmocean as cmo
import cmasher as cmr
import matplotlib.cm as cm

class Surface:

    def __init__(self, cube=None, PA=None, inc=None, x_c=None, y_c=None, v_syst=None, sigma=8, **kwargs):

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
            path = self.cube.filename
            bm_data, bm_velax = bm.load_cube(path)
            rms = bm.estimate_RMS(data=bm_data, N=1)
            self.rms = rms
            print("rms =",rms)
        
        self._detect_surface()
        #self._compute_surface()
        #self._plot_mol_surface()
        self._plot_traced_channels()

        return
    

    def _compute_velocity_fields(self):
        """
        For computing the line of sight velocity fields using the bettermoments package
        """

        path = self.cube.filename
        bm_data, bm_velax = bm.load_cube(path)

        rms = bm.estimate_RMS(data=bm_data, N=1)

        user_mask = bm.get_user_mask(data=bm_data, user_mask_path=None)
        threshold_mask = bm.get_threshold_mask(data=bm_data, clip=3.0)
        channel_mask = bm.get_channel_mask(data=bm_data, firstchannel=75, lastchannel=278)

        mask = bm.get_combined_mask(user_mask=user_mask, threshold_mask=threshold_mask, channel_mask=channel_mask, combine='and')

        masked_bm_data = bm_data * mask 
        
        moments = bm.collapse_gaussian(velax=bm_velax, data=masked_bm_data, rms=rms)

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

        # parameters
        nx, nv, ny = self.cube.nx, self.cube.nv, self.cube.ny
        v = self.cube.velocity[:,np.newaxis]
        dv = round(self.cube.velocity[1]-self.cube.velocity[0], 3)
        print("spectral res. =",dv)

        # setting up arrays
        x_surf = np.zeros([nv,nx])
        y_surf = np.zeros([nv,nx,2])
        Bv_surf = np.zeros([nv,nx,2])

        # load in velocity map
        vfields = casa.Cube(self.cube.filename.replace(".fits","_gv0.fits"))
        vfields_im = np.nan_to_num(vfields.image[:,:])
        vfields_im_rot = rotate_disc(vfields_im, PA=self.PA, x_c=self.x_c, y_c=self.y_c)

        # Loop over the channels
        for iv in range(nv):
            
            print(iv,"/",nv-1, v[iv][0])
            
            # rotate the image so major axis is aligned with x-axis.
            im = np.nan_to_num(self.cube.image[iv,:,:])
            im_rot = rotate_disc(im, PA=self.PA, x_c=self.x_c, y_c=self.y_c)

            # setting up arrays in each channel
            in_surface = np.full(nx, False)
            local_surf = np.zeros([nx,2], dtype=int)
            local_exact = np.zeros([nx,2])
            B_surf = np.zeros([nx,2])

            # loop through each x-coordinate
            for i in range(nx):

                vert_profile = im_rot[:,i]
                vfields_profile = vfields_im_rot[:,i] / 1.e3  #add a check for units and convert if necessary.
            
                # finding the flux maxima for each slice in the x-axis
                local_max, mom9_coords = search_maxima(vert_profile, vfields_profile, v=v[iv][0], dv=dv, y_c=self.y_c, threshold=self.sigma*self.rms, dx=self.cube.bmaj/self.cube.pixelscale)

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
        y_a = self.y_surf[:,:,1] - self.y_c
        y_b = self.y_surf[:,:,0] - self.y_c

        # determining which surface (top/bottom) is the near/far side.
        y_mean = np.mean(self.y_surf[:,:,:], axis=2) - self.y_c
        mask = (y_mean == 0)    # removing x coordinates with no traced points.
        y_mean_masked = np.ma.masked_array(y_mean, mask).compressed()
        
        if (len(np.where(y_mean_masked.ravel() < self.y_c)[0]) > 0.5*len(y_mean_masked.ravel())):
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
        var = [self.h, self.v, self.Tb]
        stat = ['mean', 'mean', 'max']

        freq = str(round(self.cube.restfreq/1.e9))
        source = self.cube.object
        location = os.path.dirname(os.path.realpath(self.cube.filename))
        
        for k in range(3):

            fig = plt.figure(figsize=(6,6))
            gs = gridspec.GridSpec(1,1)
            ax = plt.subplot(gs[0])
            
            data,_,_ = binned_statistic(self.r, [self.r, var[k]], statistic=stat[k], bins=bins)
            std,_,_ = binned_statistic(self.r, var[k], statistic='std', bins=bins)

            ax.scatter(data[0,:], data[1,:], alpha=0.7, s=5, label=isotope)
            ax.errorbar(data[0,:], data[1,:], yerr=std, ls='none')

            ax.set_xlabel('r [arcsec]')
            ax.set_ylabel(plot[k]+units[k])

            np.savetxt(location+'/'+source+'_'+freq+'_surface_params.txt', np.column_stack([self.r, self.h, self.v, self.Tb]))

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
        xc_arc, yc_arc, extent = self.cube._arcsec_coords(xc=self.x_c, yc=self.y_c)
        #xc_arc = -(self.x_c - ((self.cube.nx - 1) / 2)) * self.cube.pixelscale
        #yc_arc = (self.y_c - ((self.cube.ny - 1) / 2)) * self.cube.pixelscale
        
        ############
        nv = self.cube.nv

        norm = PowerNorm(1, vmin=0, vmax=np.max(self.cube._Jybeam_to_Tb(np.nan_to_num(self.cube.image[:,:,:]))))
        cmap = cmo.cm.rain

        freq = str(round(self.cube.restfreq/1.e9))
        source = self.cube.object
        location = os.path.dirname(os.path.realpath(self.cube.filename))
        output = location+'/'+source+'_'+freq+'GHz_layers.pdf'
        if os.path.exists(output):
            os.system('rm -rf output')
            
        with PdfPages(output) as pdf:
            for iv in range(nv):

                fig = plt.figure(figsize=(6,6))
                gs = gridspec.GridSpec(1,1)
                ax = plt.subplot(gs[0])
                
                im_K = self.cube._Jybeam_to_Tb(np.nan_to_num(self.cube.image[iv,:,:]))

                im = rotate_disc(im_K, PA=self.PA, x_c=self.x_c, y_c=self.y_c) 

                image = ax.imshow(im, origin='lower', cmap=cmap, norm=norm, extent=extent)

                # adding marker for disc centre
                ax.plot(xc_arc, yc_arc, '+', color='white')

                ## adding trace points                
                ax.plot(x_arc[iv,:],y_arc[iv,:,0], '.', markersize=2, color='white')
                ax.plot(x_arc[iv,:],y_arc[iv,:,1], '.', markersize=2, color='white')

                # zooming in on the surface
                '''
                # to be updated
            
                xmin = np.min(np.min(x_arc[iv]) - 0.2*(abs(np.min(x_arc[iv]) - self.x_c)), extent[3])
                xmax = np.max(np.max(x_arc[iv]) + 0.2*(abs(np.max(x_arc[iv]) - self.x_c)), extent[1])
                ymin = np.max(np.min(y_arc[iv]) - 0.2*(abs(np.min(y_arc[iv]) - self.y_c)), extent[2])
                ymax = np.min(np.max(y_arc[iv]) + 0.2*(abs(np.max(y_arc[iv]) - self.y_c)), extent[0])
            
                plt.xlim(xmin, xmax)
                plt.ylim(ymin, ymax)
            
                # adding beam
                ax = plt.gca()
                beam = Ellipse(xy=(xmin + 0.1*abs(xmax-xmin), ymax - 0.1*abs(ymax-ymin)), width=self.cube.bmin, height=self.cube.bmaj, angle=-self.cube.bpa, fill=True, color='white')
                ax.add_patch(beam)
                '''

                pdf.savefig(bbox_inches='tight')
                plt.close()


def search_maxima(yprofile, velprofile, v=None, dv=None, y_c=None, threshold=None, dx=0):

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
