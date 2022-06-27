import os
import sys
import my_casa_cube as casa

import cv2
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

#from scipy.interpolate import interp1d


class Surface:

    def __init__(self, cube=None, PA=None, inc=None, x_c=None, y_c=None, v_syst=None, sigma=5, **kwargs):
        
        self.cube = cube
        self.PA = PA
        self.inc = inc
        self.x_c = x_c
        self.y_c = y_c
        self.sigma = sigma
        self.v_syst = v_syst

        im_3D = np.copy(self.cube.image)
        dv = np.abs(self.cube.velocity[1] - self.cube.velocity[0])
        self.dv = dv / 1000
        
        rms = np.nanstd(self.cube.image[0,:,:])
        self.rms = rms

        vfields = casa.Cube(self.cube.filename.replace(".fits","_mom9.fits"))
        vfields_im = np.nan_to_num(vfields.image[0,:,:])

        self.vfields = vfields_im
        
        self._detect_surface()    
        self._plot_traced_channels()
        #self._compute_surface()
        #self._plot_mol_surface()

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
        dv = self.dv
        print("spectral res. =",dv)

        # setting up arrays
        x_surf = np.zeros([nv,nx])
        y_surf = np.zeros([nv,nx,4])
        Bv_surf = np.zeros([nv,nx,4])
        
        # Loop over the channels
        for iv in range(1): 

            iv += 40
            
            print(iv,"/",nv-1, v[iv][0])
         
            # rotate the image so major axis is aligned with x-axis.
            im = np.nan_to_num(self.cube.image[iv,:,:])
            im_rot = rotate_disc(im, PA=self.PA, x_c=self.x_c, y_c=self.y_c)
            vfields_im = np.nan_to_num(self.vfields)
            vfields_im_rot = rotate_disc(self.vfields, PA=self.PA, x_c=self.x_c, y_c=self.y_c)

            # masking per channel

            vfields_im_rot[im_rot < self.sigma*self.rms] = 0
            im_rot[im_rot < self.sigma*self.rms] = 0

            kernel = int(np.ceil(3*(self.cube.bmaj/self.cube.pixelscale)) // 2 * 2 + 1)
            im_rot_blurred = cv2.GaussianBlur(im_rot/255.0, (kernel, kernel), cv2.BORDER_DEFAULT)
            im_rot = im_rot_blurred
            '''
            plt.imshow(im_rot, origin='lower')
            plt.show()
            sys.exit()
            '''
            # setting up arrays in each channel
            in_surface = np.full(nx, False)
            local_surf = np.zeros([nx,4], dtype=int)
            B_surf = np.zeros([nx,4])

            # loop through each x-coordinate
            for i in range(nx):

                #i += 178
                
                vert_profile = im_rot[:,i]
                vfields_profile = vfields_im_rot[:,i]
            
                # finding the flux maxima for each slice in the x-axis
                local_max = search_maxima(vert_profile, vfields_profile, v=v[iv][0], dv=dv, y_c=self.y_c, ny=ny, dx=self.cube.bmaj/self.cube.pixelscale)
                #print(local_max)
                
                if any(x is not None for x in local_max) is True:

                    in_surface[i] = True

                    for k in range(4):

                        if local_max[k] is None:
                            local_surf[i,k] = 0
                        else:
                            local_surf[i,k] = local_max[k]
                            j = local_surf[i,k]
                            B_surf[i,k] = im_rot[j,i]

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

        # y-coordinates for surfaces vertically above and below disc centre in sky coordinates
        y_a = self.y_surf[:,:,1] - self.y_c
        y_b = self.y_surf[:,:,0] - self.y_c

        # determining which surface (top/bottom) is the near/far side.
        y_mean = np.mean(self.y_surf[:,:,:], axis=2) - self.y_c
        mask = (y_mean == 0)    # removing x coordinates with no traced points.
        y_mean_masked = np.ma.masked_array(y_mean, mask).compressed()
        
        if (len(np.where(y_mean_masked.ravel() > 0)[0]) > 0.5*len(y_mean_masked.ravel())):
            condition = 0
        else:
            condition = 1

        # computing the radius and height and converting units to arcseconds
        if condition == 0:
            print('bottom layer is the near side')
            h = abs(y_mean) / np.sin(inc_rad)
            r = np.hypot(self.x_surf - self.x_c, (y_a - y_mean) / np.cos(inc_rad))
        elif condition == 1:
            print('top layer is the near side')
            h = abs(y_mean) / np.sin(inc_rad)
            r = np.hypot(self.x_surf - self.x_c, abs(y_b - y_mean) / np.cos(inc_rad))

        r *= self.cube.pixelscale
        h *= self.cube.pixelscale        

        # computing velocity and brightness profile
        v = (self.cube.velocity[:,np.newaxis] - self.v_syst) * r / ((self.x_surf - self.x_c) * np.sin(inc_rad))
        Bv = np.mean(self.Bv_surf[:,:,:], axis=2)

        # masking invalid points
        mask1 = np.isinf(v) | np.isnan(v) | (h<0)

        r = np.ma.masked_array(r,mask1).compressed()
        h = np.ma.masked_array(h,mask1).compressed()
        v = np.ma.masked_array(v,mask1).compressed()
        Bv = np.ma.masked_array(Bv,mask1).compressed()

        # check is the disc is rotating in the opposite direction
        if (np.mean(v) < 0):
            v = -v 

        mask2 = (v<0)

        r = np.ma.masked_array(r,mask2).compressed()
        h = np.ma.masked_array(h,mask2).compressed()
        v = np.ma.masked_array(v,mask2).compressed()
        Bv = np.ma.masked_array(Bv,mask2).compressed()

        # compute brightness temperature
        Tb = self.cube._Jybeam_to_Tb(Bv)

        self.r = r
        self.h = h
        self.v = v
        self.Tb = Tb


    def _plot_mol_surface(self):
        """
        Plotting surfaces
        """
        
        bins = 100
        plot = ['h', 'v', 'Tb']
        units = ['[arcsec]', '[km/s]', '[K]']
        var = [self.h, self.v, self.Tb]
        stat = ['mean', 'mean', 'max']

        freq = str(round(self.cube.restfreq/1.e9))
        source = self.cube.object
        location = os.path.dirname(os.path.realpath(self.cube.filename))
        output = location+'/'+source+'_'+freq+'GHz_surfaces.pdf'
        if os.path.exists(output):
            os.system('rm -rf output')

        # saving traces
        np.savetxt(location+'/'+source+'_'+freq+'GHz_surface_params.txt', np.column_stack([self.r, self.h, self.v, self.Tb]))

        # plotting surfaces
        with PdfPages(output) as pdf:
            for k in range(3):

                fig = plt.figure(figsize=(6,6))
                gs = gridspec.GridSpec(1,1)
                ax = plt.subplot(gs[0])
            
                data,_,_ = binned_statistic(self.r, [self.r, var[k]], statistic=stat[k], bins=bins)
                std,_,_ = binned_statistic(self.r, var[k], statistic='std', bins=bins)

                ax.scatter(data[0,:], data[1,:], alpha=0.7, s=5)
                ax.errorbar(data[0,:], data[1,:], yerr=std, ls='none')

                ax.set_xlabel('r [arcsec]')
                ax.set_ylabel(plot[k]+units[k])
                
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
                ax.plot(x_arc[iv,:],y_arc[iv,:,0], 'o', color='blue')
                ax.plot(x_arc[iv,:],y_arc[iv,:,1], '+', color='orange')
                #ax.plot(x_arc[iv,:],y_arc[iv,:,2], '.', color='red')
                #ax.plot(x_arc[iv,:],y_arc[iv,:,3], '.', color='pink')
                
            
                # adding beam
                #ax = plt.gca()
                #beam = Ellipse(xy=(xmax - 3*self.cube.bmaj/self.cube.pixelscale, ymin + 3*self.cube.bmaj/self.cube.pixelscale), width=self.cube.bmin, height=self.cube.bmaj, angle=-self.cube.bpa, fill=True, color='white')
                #ax.add_patch(beam)

                pdf.savefig(bbox_inches='tight')
                plt.close()


def search_maxima(line_profile, v_profile, v=None, dv=None, y_c=None, ny=None, dx=0):

    # find local maxima
    #v += 0.5 * dv
    
    dI = line_profile[1:] - line_profile[:-1]
    I_max = np.where((np.hstack((0, dI)) > 0) & (np.hstack((dI, 0)) < 0))[0]

    # sort from strongest signal to weakest
    
    I_max = I_max[np.argsort(line_profile[I_max])][::-1]
    
    # finding iso-velocity curves

    v_diff = abs(v_profile - v)
    
    v_idx = np.where(v_diff < dv)[0]
    
    if v_idx.size >= 2:

        print(v_idx)
        filtered_v_idx = []
        filtered_v_idx.append(v_idx[0])
        previous = v_idx[0]
        for i in range(1, v_idx.size):
            diff = v_idx[i] - previous
            if diff > dx:
                filtered_v_idx.append(v_idx[i])
                previous = v_idx[i]

        v_idx = np.array(filtered_v_idx)
     
        v_idx = v_idx[np.argsort(line_profile[v_idx])][::-1]

        print(v_idx)
           
    # determining the surfaces

    #if I_max.size >= 2 and v_idx.size >= 2:

        
    
    v_iso = v_idx[:2]
    
    if len(I_max) and len(v_iso) >= 2:
        
        coord_1 = np.max(v_iso)
        coord_2 = np.min(v_iso)

        '''
        front_surface_upper = next(iter(I_max[np.where(abs(I_max - coord_1) < dx)]), None)        
        front_surface_lower = next(iter(I_max[np.where(abs(I_max - coord_2) < dx)]), None)

        if front_surface_upper is None or front_surface_lower is None:
            back_surface_upper = None
            back_surface_lower = None

        if front_surface_upper is not None and front_surface_lower is not None:
            if abs(front_surface_upper - front_surface_lower) < dx:
                back_surface_upper = None
                back_surface_lower = None
        '''  
        front_surface_upper = coord_1
        front_surface_lower = coord_2
        
        back_surface_upper = None
        back_surface_lower = None
        
    else:

        front_surface_upper = None
        front_surface_lower = None
        back_surface_upper = None
        back_surface_lower = None
        
    '''
    elif len(I_max) and len(v_diff) == 3:

        coord_1 = v_iso[0]
        coord_2 = v_iso[1]
        coord_3 = v_iso[2]

        if coord_1 > coord_2:
            
            front_surface_upper = I_max[np.where(abs(I_max - coord_1) < dx)]
            if len(front_surface_upper) > 0:
                front_surface_upper = front_surface_upper[0]
            else:
                front_surface_upper = None
                
            front_surface_lower = I_max[np.where(abs(I_max - coord_2) < dx)]
            if len(front_surface_lower) > 0:
                front_surface_lower = front_surface_lower[0]
            else:
                front_surface_lower = None
                
        else:
            
            front_surface_upper = I_max[np.where(abs(I_max - coord_2) < dx)]
            if len(front_surface_upper) > 0:
                front_surface_upper = front_surface_upper[0]
            else:
                front_surface_upper = None
                
            front_surface_lower = I_max[np.where(abs(I_max - coord_1) < dx)]
            if len(front_surface_lower) > 0:
                front_surface_lower = front_surface_lower[0]
            else:
                front_surface_lower = None
            
        if coord_3 > y_c:
            
            back_surface_upper = I_max[np.where(abs(I_max - coord_3) < dx)]
            if len(front_surface_upper) > 0:
                back_surface_upper = front_surface_upper[0]
            else:
                back_surface_upper = None
            back_surface_lower = None
            
        else:
            
            back_surface_upper = None
            back_surface_lower = I_max[np.where(abs(I_max - coord_3) < dx)]
            if len(front_surface_lower) > 0:
                back_surface_lower = front_surface_lower[0]
            else:
                back_surface_lower = None
        
    elif len(I_max) and len(v_diff) >= 4:
        
        coord_1 = v_iso[0]
        coord_2 = v_iso[1]
        coord_3 = v_iso[2]
        coord_4 = v_iso[3]

        if coord_1 > coord_2:
            
            front_surface_upper = I_max[np.where(abs(I_max - coord_1) < dx)]
            if len(front_surface_upper) > 0:
                front_surface_upper = front_surface_upper[0]
            else:
                front_surface_upper = None
                
            front_surface_lower = I_max[np.where(abs(I_max - coord_2) < dx)]
            if len(front_surface_lower) > 0:
                front_surface_lower = front_surface_lower[0]
            else:
                front_surface_lower = None
                
        else:
            
            front_surface_upper = I_max[np.where(abs(I_max - coord_2) < dx)]
            if len(front_surface_upper) > 0:
                front_surface_upper = front_surface_upper[0]
            else:
                front_surface_upper = None
                
            front_surface_lower = I_max[np.where(abs(I_max - coord_1) < dx)]
            if len(front_surface_lower) > 0:
                front_surface_lower = front_surface_lower[0]
            else:
                front_surface_lower = None
            
        if coord_3 > coord_4:
            
            back_surface_upper = I_max[np.where(abs(I_max - coord_3) < dx)]
            if len(back_surface_upper) > 0:
                back_surface_upper = back_surface_upper[0]
            else:
                back_surface_upper = None
                
            back_surface_lower = I_max[np.where(abs(I_max - coord_4) < dx)]
            if len(back_surface_lower) > 0:
                back_surface_lower = back_surface_lower[0]
            else:
                back_surface_lower = None
                
        else:
            
            back_surface_lower = I_max[np.where(abs(I_max - coord_4) < dx)]
            if len(back_surface_lower) > 0:
                back_surface_lower = back_surface_lower[0]
            else:
                back_surface_lower = None
                
            back_surface_upper = I_max[np.where(abs(I_max - coord_3) < dx)]
            if len(back_surface_upper) > 0:
                back_surface_upper = back_surface_upper[0]
            else:
                back_surface_upper = None
    
    else:

        front_surface_upper = None
        front_surface_lower = None
        back_surface_upper = None
        back_surface_lower = None
    '''       
    
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
