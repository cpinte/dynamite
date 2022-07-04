import os
import sys
import my_casa_cube as casa

import cv2
from scipy.signal import find_peaks
import numpy as np
import scipy.constants as sc
from scipy import ndimage
from scipy.ndimage import rotate
from scipy.stats import binned_statistic
from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cmocean as cmo
import cmasher as cmr
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from matplotlib.colors import LogNorm, PowerNorm
from alive_progress import alive_bar
from time import sleep

np.set_printoptions(threshold=np.inf)
    
class Surface:

    def __init__(self, cube=None, PA=None, inc=None, x_c=None, y_c=None, v_syst=None, sigma=5, **kwargs):
        
        self.cube = cube
        self.PA = PA
        self.inc = inc
        self.x_c = x_c
        self.y_c = y_c
        self.sigma = sigma
        self.v_syst = v_syst
        
        rms = np.nanstd(self.cube.image[0,:,:])
        self.rms = rms
        print('rms =', rms)
        
        self._detect_surface()    
        self._plot_traced_channels()
        self._compute_surface()
        self._plot_mol_surface()

        return
        
        
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
        v = self.cube.velocity[:,np.newaxis] / 1000

        # setting up arrays
        x_surf = np.zeros([nv,nx])
        y_surf = np.zeros([nv,nx,4])
        Bv_surf = np.zeros([nv,nx,4])
        
        # Loop over the channels

        with alive_bar(nv) as bar:
            for iv in range(nv):

                #iv += 14
         
                # rotate the image so major axis is aligned with x-axis.
                im = np.nan_to_num(self.cube.image[iv,:,:])
                im_rot = rotate_disc(im, PA=self.PA, x_c=self.x_c, y_c=self.y_c)

                # masking per channel
                
                # setting up arrays in each channel
                in_surface = np.full(nx, False)
                local_surf = np.zeros([nx,4], dtype=int)
                B_surf = np.zeros([nx,4])

                dx=self.cube.bmaj/self.cube.pixelscale
            
                # loop through each x-coordinate
                for i in range(nx):
                
                    vert_profile = im_rot[:,i]
                    #vfields_profile = vfields_im_rot[:,i]
            
                    # finding the flux maxima for each slice in the x-axis
                    local_max = search_maxima(vert_profile, dx=dx, threshold=self.sigma*self.rms, rms=self.rms)
                  
                    if any(x is not None for x in local_max) is True:

                        for k in range(4):

                            if local_max[k] is None:
                                local_surf[i,k] = 0
                            else:
                                local_surf[i,k] = local_max[k]
                                j = local_surf[i,k]
                                B_surf[i,k] = im_rot[j,i]
                
                # finding continuous points using volatility
                
                for k in range(4):

                    pre_coord = None
                    pre_grad = None
                    coord = local_surf[:,k]
                    x_old = None

                    for j in range(coord.size):
                    
                        if v[iv] < self.v_syst:
                            j = j
                        else:
                            j = abs(j - (coord.size - 1))
                        
                        if pre_coord is not None and x_old is not None and coord[j] != 0:
                            grad = coord[j] / pre_coord    
                        
                            if pre_grad is not None:
                                volatility = np.std([np.log(grad),np.log(pre_grad)])
                            
                                if volatility > 0.05:
                                    local_surf[j,k] = 0
                                    B_surf[j,k] = 0
                                else:
                                    pre_grad = grad
                                    pre_coord = coord[j]
                                    x_old = j
                            else:
                                pre_grad = grad
                        
                        if pre_coord is None and x_old is None and coord[j] != 0:
                            pre_coord = coord[j]
                            x_old = j
                
                # removing traces without upper and lower surfaces
                
                for j in range(local_surf[:,0].size):
                    if local_surf[j,0] == 0 or local_surf[j,1] == 0:
                        local_surf[j,0] = 0
                        local_surf[j,1] = 0
                    if local_surf[j,2] == 0 or local_surf[j,3] == 0:
                        local_surf[j,2] = 0
                        local_surf[j,3] = 0
                
                # for only storing slices with surfaces
            
                for i in range(nx):
                    
                    if any(x is not None for x in local_surf[i,:]) is True:
                        in_surface[i] = True

                # writing to 3D arrays
            
                if np.any(in_surface):
            
                    x = np.arange(nx)            
                    n = np.sum(in_surface)
                    if n > 0:
                        x_surf[iv,:n] = x[in_surface]
                        y_surf[iv,:n,:] = local_surf[in_surface,:]
                        Bv_surf[iv,:n,:] = B_surf[in_surface,:]

                # progress bar
                sleep(0.001)
                bar()

            # saving globally and in text file
            
            self.x_surf = x_surf
            self.y_surf = y_surf
            self.Bv_surf = Bv_surf
                
            freq = str(round(self.cube.restfreq/1.e9))
            source = self.cube.object
            location = os.path.dirname(os.path.realpath(self.cube.filename))
            np.savetxt(location+'/'+source+'_'+freq+'GHz_pixel_coords.txt', np.column_stack([self.x_surf, self.y_surf[:,:,0], self.y_surf[:,:,1]]))


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

        # FRONT SURFACE
        h_front = abs(np.mean(self.y_surf[:,:,:2], axis=2) - self.y_c) / np.sin(inc_rad)
        r_front = np.hypot(self.x_surf - self.x_c, (self.y_surf[:,:,1] - np.mean(self.y_surf[:,:,:2], axis=2)) / np.cos(inc_rad))
        v_front = (self.cube.velocity[:,np.newaxis] - self.v_syst) * r_front / ((self.x_surf - self.x_c) * np.sin(inc_rad))
        Bv_front = np.mean(self.Bv_surf[:,:,:2], axis=2)

        # checking rotation of disc
        if (np.nanmean(v_front[~np.isinf(v_front)])) < 0:
            v_front *= -1

        mask = np.isinf(v_front) | np.isnan(v_front) | (h_front<0) | (v_front<0)

        h_front = np.ma.masked_array(h_front,mask).compressed()
        r_front = np.ma.masked_array(r_front,mask).compressed()
        v_front = np.ma.masked_array(v_front,mask).compressed()
        Bv_front = np.ma.masked_array(Bv_front,mask).compressed()        

        # compute brightness temperature
        Tb_front = self.cube._Jybeam_to_Tb(Bv_front)
        

        # BACK SURFACE
        h_back = abs(np.mean(self.y_surf[:,:,2:4], axis=2) - self.y_c) / np.sin(inc_rad)
        r_back = np.hypot(self.x_surf - self.x_c, (self.y_surf[:,:,3] - np.mean(self.y_surf[:,:,2:4], axis=2)) / np.cos(inc_rad))
        v_back = (self.cube.velocity[:,np.newaxis] - self.v_syst) * r_back / ((self.x_surf - self.x_c) * np.sin(inc_rad))
        Bv_back = np.mean(self.Bv_surf[:,:,2:4], axis=2)

        # checking rotation of disc
        if (np.nanmean(v_back[~np.isinf(v_back)])) < 0:
            v_back *= -1

        mask = np.isinf(v_back) | np.isnan(v_back) | (h_back<0) | (v_back<0)

        h_back = np.ma.masked_array(h_back,mask).compressed()
        r_back = np.ma.masked_array(r_back,mask).compressed()
        v_back = np.ma.masked_array(v_back,mask).compressed()
        Bv_back = np.ma.masked_array(Bv_back,mask).compressed()        

        # compute brightness temperature
        Tb_back = self.cube._Jybeam_to_Tb(Bv_back)

        self.r_front = r_front
        self.h_front = h_front
        self.v_front = v_front
        self.Tb_front = Tb_front

        self.r_back = r_back
        self.h_back = h_back
        self.v_back = v_back
        self.Tb_back = Tb_back


    def _plot_mol_surface(self):
        """
        Plotting surfaces
        """

        # output file
        freq = str(round(self.cube.restfreq/1.e9))
        source = self.cube.object
        location = os.path.dirname(os.path.realpath(self.cube.filename))
        output = location+'/'+source+'_'+freq+'GHz_surfaces.pdf'
        if os.path.exists(output):
            os.system('rm -rf output')

        # setting up arrays
        bins = 100
        plot = ['h', 'v', 'Tb']
        units = ['[arcsec]', '[km/s]', '[K]']
        var_front = [self.h_front, self.v_front, self.Tb_front]
        var_back = [self.h_back, self.v_back, self.Tb_back]
        stat = ['mean', 'mean', 'max']

        # saving traces
        np.savetxt(location+'/'+source+'_'+freq+'GHz_surface_params.txt', np.column_stack([self.r_front, self.h_front, self.v_front, self.Tb_front]))

        # plotting surfaces
        with PdfPages(output) as pdf:
            for k in range(3):

                fig = plt.figure(figsize=(6,6))
                gs = gridspec.GridSpec(1,1)
                ax = plt.subplot(gs[0])
                
                #ax.plot(self.r_front, var_front[k], 'o', color='blue')
                #ax.plot(self.r_back, var_back[k], '.', color='orange')
                
                for i in range(2):

                    if i == 0:
                        r = self.r_front
                        var = var_front
                        label = 'front surface'
                    elif i == 1:
                        r = self.r_back
                        var = var_back
                        label = 'back surface'
                        
                    data,_,_ = binned_statistic(r, [r, var[k]], statistic=stat[k], bins=bins)
                    std,_,_ = binned_statistic(r, var[k], statistic='std', bins=bins)
                    ax.scatter(data[0,:], data[1,:], alpha=0.7, s=5, label=label)
                    ax.errorbar(data[0,:], data[1,:], yerr=std, ls='none')
                
                ax.set_xlabel('r [arcsec]')
                ax.set_ylabel(plot[k]+units[k])
                ax.legend(loc='upper right')
                
                pdf.savefig(bbox_inches='tight')
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
        
        # finding plotting limits
        '''
        xmin = np.nanmax([np.nanmin(x_arc[:,:]) - 0.2*(abs(np.nanmin(x_arc[:,:]) - xc_arc)), extent[1]])
        xmax = np.nanmin([np.nanmax(x_arc[:,:]) + 0.2*(abs(np.nanmax(x_arc[:,:]) - xc_arc)), extent[0]])
        ymin = np.nanmax([np.nanmin(y_arc[:,:,0]) - 0.2*(abs(np.nanmin(y_arc[:,:,0]) - yc_arc)), extent[2]])
        ymax = np.nanmin([np.nanmax(y_arc[:,:,1]) + 0.2*(abs(np.nanmax(y_arc[:,:,1]) - yc_arc)), extent[3]])
        '''
        
        ############
        nv = self.cube.nv

        norm = PowerNorm(1, vmin=0, vmax=np.max(self.cube._Jybeam_to_Tb(np.nan_to_num(self.cube.image[:,:,:]))))
        cmap = cm.Greys_r

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

                # zooming in on the surface
                #plt.xlim(xmax, xmin)
                #plt.ylim(ymin, ymax)
                #plt.xlim(1.5, -1.5)
                #plt.ylim(-1.5, 1.5)
                
                ## adding trace points                
                ax.plot(x_arc[iv,:],y_arc[iv,:,0], '.', color='blue', label='front upper')
                ax.plot(x_arc[iv,:],y_arc[iv,:,1], '.', color='orange', label='front lower')
                ax.plot(x_arc[iv,:],y_arc[iv,:,2], '.', color='red', label='back upper')
                ax.plot(x_arc[iv,:],y_arc[iv,:,3], '.', color='green', label='back lower')

                plt.legend()
                
                # adding beam
                #ax = plt.gca()
                #beam = Ellipse(xy=(xmax - 3*self.cube.bmaj/self.cube.pixelscale, ymin + 3*self.cube.bmaj/self.cube.pixelscale), width=self.cube.bmin, height=self.cube.bmaj, angle=-self.cube.bpa, fill=True, color='white')
                #ax.add_patch(beam)

                pdf.savefig(bbox_inches='tight')
                plt.close()
                

def search_maxima(line_profile, dx=0, threshold=0, rms=0):

    # find local maxima

    peaks, _ = find_peaks(line_profile, distance = dx, width = 0.5*dx, height = threshold, prominence = 2 * rms)

    # determining the surfaces

    peaks_sorted = peaks[np.argsort(line_profile[peaks])][::-1]
    
    if peaks.size >= 2 and peaks.size < 4:

        front_surface = peaks_sorted[:2]
        front_surface_upper = np.max(front_surface)
        front_surface_lower = np.min(front_surface)

        back_surface_upper = None
        back_surface_lower = None

    elif peaks.size >= 4:
        
        front_surface = peaks_sorted[:2]
        front_surface_upper = np.max(front_surface)
        front_surface_lower = np.min(front_surface)

        back_surface = peaks_sorted[2:4]
        back_surface_upper = np.max(back_surface)
        back_surface_lower = np.min(back_surface)      
        
    else:
        
        front_surface_upper = None
        front_surface_lower = None
        back_surface_upper = None
        back_surface_lower = None
    
    coords = [front_surface_upper, front_surface_lower, back_surface_upper, back_surface_lower]

    return coords
    

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
