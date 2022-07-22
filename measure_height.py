import os
import sys
import my_casa_cube as casa

import cv2
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import numpy as np
import scipy.constants as sc
from scipy import ndimage
from scipy.ndimage import rotate
from skimage.transform import resize
from scipy.stats import binned_statistic
from matplotlib.backends.backend_pdf import PdfPages
from astropy.io import fits

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

    def __init__(self, data=None, PA=None, inc=None, x_c=None, y_c=None, v_syst=None, distance=None, sigma=5, **kwargs):
        
        self.cube = Cube(data)
        self.PA = PA
        self.inc = inc
        self.x_c = x_c
        self.y_c = y_c
        self.sigma = sigma
        self.v_syst = v_syst
        self.distance = distance
        
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
        extract the emission surface in each channel and loop over channels.
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
         
                # rotate the image so major axis is aligned with x-axis.
                
                im = np.nan_to_num(self.cube.image[iv,:,:])
                im_rot = rotate_disc(im, PA=self.PA, x_c=self.x_c, y_c=self.y_c)
                
                # setting up arrays in each channel
        
                in_surface = np.full(nx, False)
                local_surf = np.zeros([nx,4], dtype=int)
                B_surf = np.zeros([nx,4])

                dx=self.cube.bmaj/self.cube.pixelscale
            
                # looping through each x-coordinate
                
                for i in range(nx):
                
                    vert_profile = im_rot[:,i]
            
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
                
                # removing discontinuous outliers 
                
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
                
                # removing slices without both points on both the upper and lower surfaces 
                
                for j in range(local_surf[:,0].size):
                    if local_surf[j,0] == 0 or local_surf[j,1] == 0:
                        local_surf[j,0] = 0
                        local_surf[j,1] = 0
                    if local_surf[j,2] == 0 or local_surf[j,3] == 0:
                        local_surf[j,2] = 0
                        local_surf[j,3] = 0
                
                # only storing channels with valid traces
            
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

            # saving traces
            
            self.x_surf = x_surf
            self.y_surf = y_surf
            self.Bv_surf = Bv_surf
                
            np.savetxt(self.cube.filename+'_trace_pixel_coordinates.txt', np.column_stack([self.x_surf, self.y_surf[:,:,0], self.y_surf[:,:,1]]))


    def _compute_surface(self):
        """
        For estimating r, h, and v of the emitting layer.

        """

        inc_rad = np.radians(self.inc)

        # FRONT SURFACE
        h_front = abs(np.mean(self.y_surf[:,:,:2], axis=2) - self.y_c) / np.sin(inc_rad)
        h_front *= self.cube.pixelscale
        r_front = np.hypot(self.x_surf - self.x_c, (self.y_surf[:,:,1] - np.mean(self.y_surf[:,:,:2], axis=2)) / np.cos(inc_rad))
        r_front *= self.cube.pixelscale 
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
        h_back *= self.cube.pixelscale
        r_back = np.hypot(self.x_surf - self.x_c, (self.y_surf[:,:,3] - np.mean(self.y_surf[:,:,2:4], axis=2)) / np.cos(inc_rad))
        r_back *= self.cube.pixelscale 
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
        
        output = self.cube.filename+'_surfaces.pdf'
        if os.path.exists(output):
            os.system('rm -rf output')

        # setting up arrays
        
        plot = ['h', 'v', 'Tb']
        units = ['[arcsec]', '[km/s]', '[K]']
        var_front = [self.h_front, self.v_front, self.Tb_front]
        var_back = [self.h_back, self.v_back, self.Tb_back]

        # saving traces
        
        np.savetxt(self.cube.filename+'_surface_params.txt', np.column_stack([self.r_front, self.h_front, self.v_front, self.Tb_front]))

        # plotting surfaces
        with PdfPages(output) as pdf:
            for k in range(3):

                fig = plt.figure(figsize=(6,6))
                gs = gridspec.GridSpec(1,1)
                ax = plt.subplot(gs[0])

                ax.plot(self.r_front, var_front[k], '.', markersize=1, label='front surface', color='blue')
                ax.plot(self.r_back, var_back[k], '.', markersize=1, label='back surface', color='gold')
                
                if k == 0:
                        
                    # power law fitting for emission height

                    print('front surface')
                    param_front, param_cov_front = curve_fit(power_law, self.r_front * self.distance, var_front[k] * self.distance)
                    print('z0 = ', param_front[0])
                    print('p = ', param_front[1])
                    z_fit_front = (param_front[0] * (np.sort(self.r_front * self.distance) / 100)**param_front[1]) / self.distance
                    plt.plot(np.sort(self.r_front), z_fit_front, '-', markersize=1, color='blue', label='front surface - power law fit')

                    print('back surface')
                    param_back, param_cov_back = curve_fit(power_law, self.r_back * self.distance, var_back[k] * self.distance)
                    print('z0 = ', param_back[0])
                    print('p = ', param_back[1])
                    z_fit_back = (param_back[0] * (np.sort(self.r_back * self.distance) / 100)**param_back[1]) / self.distance
                    plt.plot(np.sort(self.r_back), z_fit_back, '-', markersize=1, color='gold', label='back surface - power law fit')
                    
                ax.set_xlabel('r [arcsec]')
                ax.set_ylabel(plot[k]+units[k])
                ax.legend(loc='upper right', fontsize='x-small', numpoints=3)
                
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

        output = self.cube.filename+'_layers.pdf'
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

def power_law(x, a, b):
    return a * (x/100)**b


### CASA CUBE - for reading fits file #####

FWHM_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2)))
arcsec = np.pi / 648000

class Cube:
    def __init__(self, filename, **kwargs):

        self.filename = os.path.normpath(os.path.expanduser(filename))
        self._read(**kwargs)

    def _read(self):
        try:
            hdu = fits.open(self.filename)
            self.header = hdu[0].header

            # Read a few keywords in header
            try:
                self.object = hdu[0].header['OBJECT']
            except:
                self.object = ""
            self.unit = hdu[0].header['BUNIT']

            # pixel info
            self.nx = hdu[0].header['NAXIS1']
            self.ny = hdu[0].header['NAXIS2']
            self.pixelscale = hdu[0].header['CDELT2'] * 3600 # arcsec
            self.cx = hdu[0].header['CRPIX1']
            self.cy = hdu[0].header['CRPIX2']
            self.x_ref = hdu[0].header['CRVAL1']  # coordinate
            self.y_ref = hdu[0].header['CRVAL2']
            self.FOV = np.maximum(self.nx, self.ny) * self.pixelscale

            # velocity axis
            try:
                self.nv = hdu[0].header['NAXIS3']
            except:
                self.nv = 1
            try:
                self.restfreq = hdu[0].header['RESTFRQ']
            except:
                self.restfreq = hdu[0].header['RESTFREQ']  # gildas format
            self.wl = sc.c / self.restfreq
            try:
                self.velocity_type = hdu[0].header['CTYPE3']
                self.CRPIX3 = hdu[0].header['CRPIX3']
                self.CRVAL3 = hdu[0].header['CRVAL3']
                self.CDELT3 = hdu[0].header['CDELT3']
                if self.velocity_type == "VELO-LSR": # gildas
                    self.velocity = self.CRVAL3 + self.CDELT3 * (np.arange(1, self.nv + 1) - self.CRPIX3)
                    self.nu = self.restfreq * (1 - self.velocity * 1000 / sc.c)
                elif self.velocity_type == "VRAD":  # casa format : v en km/s
                    self.velocity = (self.CRVAL3 + self.CDELT3 * (np.arange(1, self.nv + 1) - self.CRPIX3)) / 1000
                    self.nu = self.restfreq * (1 - self.velocity * 1000 / sc.c)
                elif self.velocity_type == "FREQ": # Hz
                    self.nu = self.CRVAL3 + self.CDELT3 * (np.arange(1, self.nv + 1) - self.CRPIX3)
                    self.velocity = (-(self.nu - self.restfreq) / self.restfreq * sc.c / 1000.0)  # km/s
                else:
                    raise ValueError("Velocity type is not recognised:", self.velocity_type)
            except:
                pass

            # beam
            try:
                self.bmaj = hdu[0].header['BMAJ'] * 3600 # arcsec
                self.bmin = hdu[0].header['BMIN'] * 3600
                self.bpa = hdu[0].header['BPA']
            except:
                # make an average of all the records ...
                self.bmaj = hdu[1].data[0][0]
                self.bmin = hdu[1].data[0][1]
                self.bpa = hdu[1].data[0][2]

            # reading data
            self.image = np.ma.masked_array(hdu[0].data)

            if self.image.ndim == 4:
                self.image = self.image[0, :, :, :]            

            hdu.close()
        except OSError:
            print('cannot open', self.filename)
            return ValueError


    # -- Convert disc centre coordinates into arcseconds
    def _arcsec_coords(self, downsample=None, xc=None, yc=None):

        if downsample is None:
            pixelscale = self.pixelscale
        else:
            pixelscale = downsample * self.pixelscale
            
        xc_arc = -(xc - ((self.nx - 1) / 2)) * pixelscale
        yc_arc = (yc - ((self.ny - 1) / 2)) * pixelscale

        halfsize = np.asarray([self.nx,self.ny])/2 * pixelscale
        extent = [halfsize[0], -halfsize[0], -halfsize[1], halfsize[1]]
        extent -= np.asarray([xc_arc,xc_arc,yc_arc,yc_arc])

        return xc_arc, yc_arc, extent
        
    
    # -- Write to a fits file
    def writeto(self,filename, **kwargs):
        fits.writeto(os.path.normpath(os.path.expanduser(filename)),self.image.data, self.header, **kwargs)

    # -- Functions to deal with the synthesized beam.
    def _beam_area(self):
        """Beam area in arcsec^2"""
        return np.pi * self.bmaj * self.bmin / (4.0 * np.log(2.0))

    def _beam_area_str(self):
        """Beam area in steradian^2"""
        return self._beam_area() * arcsec ** 2

    def _pixel_area(self):
        return self.pixelscale ** 2

    def _beam_area_pix(self):
        """Beam area in pix^2."""
        return self._beam_area() / self._pixel_area()

    @property
    def beam(self):
        """Returns the beam parameters in ["], ["], [deg]."""
        return self.bmaj, self.bmin, self.bpa

    def _Jybeam_to_Tb(self, im):
        """Convert flux converted from Jy/beam to K using full Planck law."""
        im2 = np.nan_to_num(im)
        nu = self.restfreq

        exp_m1 = 1e26 * self._beam_area_str() * 2.0 * sc.h * nu ** 3 / (sc.c ** 2 * abs(im2))

        hnu_kT = np.log1p(exp_m1 + 1e-10)
        Tb = sc.h * nu / (sc.k * hnu_kT)

        return np.ma.where(im2 >= 0.0, Tb, -Tb)
