# todo : iteractive procedure to fit proper maxima (not maxima along x)
# - after 1extraction, fit surface with GP
# - get isovelocity curve (could also try without by taking points before and after)
# - find maxima along the perpendicular of the isovelocity curve
# - iterate
# - after some iterations, ignore points that are far from the isovelocity curve


#todo : cleaning extraction
# - ignore after a jump in x > 1 ??
# - ignore after a jump in y > beam ?

import astropy.constants as ac
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc
from alive_progress import alive_bar
from numpy import ndarray
from scipy import ndimage
from scipy.ndimage import rotate
from scipy.optimize import curve_fit, minimize
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.stats import binned_statistic
from astropy.convolution import Gaussian2DKernel, convolve, convolve_fft
import celerite
from celerite import terms
from scipy import signal


import casa_cube

sigma_to_FWHM = 2.0 * np.sqrt(2.0 * np.log(2))
FWHM_to_sigma = 1.0 / sigma_to_FWHM
arcsec = np.pi / 648000

class Surface:

    def __init__(self,
                 cube: None,
                 PA: float = None,
                 inc: float = None,
                 dRA: float = None,
                 dDec: float = None,
                 x_star: float = None,
                 y_star: float = None,
                 v_syst: float = None,
                 sigma: float = 10,
                 dist: float = None,
                 exclude_inner_beam: bool = False,
                 num: int = 0,
                 plot: bool = True,
                 std: float = None,
                 vmin: float = None,
                 vmax: float = None,
                 iv_min: int = None,
                 iv_max: int = None,
                 no_scales: bool = False,
                 scales = None,
                 only_guess: bool = False,
                 **kwargs):
        """
        Parameters
        ----------
        cube
            Instance of an image cube of the line data open with casa_cube, or string with path to fits cube
        PA
            Position angle of the source, measured from east to north [deg]. If None, it is calculated
        inc
            Inclination of the source [deg]. If None, it is calculated
        dRA
            Offset from center in RA [arcsec]
        dDec
            Offset from center in Dec [arcsec]
        x_star
            Star position [pixels], with (nx-1)/2 as the center. If None, it is calculated
        y_star
            Star position [pixels], with (nx-1)/2 as the center. If None, it is calculated
        v_syst
            System velocity [km/s]. If None, it is calculated
        sigma
            Cut off threshold to fit surface. Default is 10
        dist
            Distance to the source [pc] for fitting stellar mass
        exclude_inner_beam
            Exclude an inner radii equal to the beam size. False by default
        plot
            Whether to display the traces and extracted height, velocity and temperature plots
        std
            Standard deviation per channel. If None, it is calculated with numpy

        Returns
        -------
        An instance with the detected surface.

        Notes
        -----
        Any notes?

        Other Parameters
        ----------------

        Examples
        --------

        """

        if isinstance(cube,str):
            print("Reading cube: "+cube)
            cube = casa_cube.Cube(cube)

        # Truncating the cube is velocity if needed
        if vmin is not None:
            iv_min = np.argmin(np.abs(cube.velocity - vmin))
        if vmax is not None:
            iv_max = np.argmin(np.abs(cube.velocity - vmax))

        if iv_min is not None:
            if iv_max is None:
                raise ValueError("iv_max is required if iv_min is provided")


            # if the velocity axis is flipped
            if iv_min > iv_max:
                temp = iv_min
                iv_min = iv_max
                iv_max = temp

            print("Truncating cube in velocity")
            cube.image = cube.image[iv_min:iv_max+1,:,:]
            cube.velocity = cube.velocity[iv_min:iv_max+1]
            cube.nv = iv_max-iv_min+1

            # Updating header
            if cube.header['CRPIX3']-1 >= iv_min: # we keep the same pixel, and adjust its index
                cube.header['CRPIX3'] -= iv_min
            else: # we use the first pixel, and update its coordinates
                cube.header['CRVAL3'] += (iv_min - (cube.header['CRPIX3']-1)) * cube.header['CDELT3']
                cube.header['CRPIX3'] = 1
                cube.header['NAXIS3'] = iv_max - iv_min

        self.cube = cube

        print("Beam", cube.bmaj, '" x ',cube.bmin,'" at PA=',cube.bpa,"deg, dv=",cube.velocity[1]-cube.velocity[0],"km/s" )

        self.sigma = sigma

        self._initial_guess(num=num,std=std)

        self.exclude_inner_beam = exclude_inner_beam

        if PA is not None:
            print("Forcing PA to:", PA)
            self.PA = PA

        if v_syst is not None:
            print("Forcing v_syst to:", v_syst)
            self.v_syst = v_syst

        if x_star is not None:
            print("Forcing star position to: ", x_star, y_star, "(pixels)")
            self.x_star = x_star
            self.y_star = y_star

        if dRA is not None:
            print("Forcing star offset to: ", dRA, dDec, "(arcsec)")
            self.x_star = (self.cube.nx-1)/2 +  dRA/self.cube.pixelscale
            self.y_star = (self.cube.ny-1)/2 + dDec/self.cube.pixelscale

        if inc is not None:
            print("Forcing inclination to ", inc, " deg")
            self.inc = inc

        if (only_guess):
            print("Exiting: only performing initial guess ")
            return

        # This is where the actual work happens
        self._rotate_cube()

        if no_scales:
            self.n_scales=1
            self.scales = cube.bmin
        else:
            if scales is None:
                self._select_scales(num=num)
            else:
                self.scales = scales
                self.n_scales = len([scales])
        print("Using ", self.n_scales, " scales")
        print("Scales are ", self.scales, " arcsec")

        self._create_rotated_cube()
        self._make_multiscale_cube()

        self._extract_isovelocity()

        # Making plots
        if plot:
            self.plot_channels(num=num+9)

        if inc is None:
            print("Estimating inclination:")
            self.find_i(num)

        self._compute_surface()

        if dist is not None:
            self.dist = dist
            self.fit_central_mass(dist=dist)
            if plot:
                self.plot_surfaces(num=num+10,m_star=self.m_star,dist=dist)
        else:
            if plot:
                self.plot_surfaces(num=num+10)

        return

    def cutout(self,FOV=None):

        new_file=self.cube.filename.replace(".fits","_cutout.fits")
        iv_buffer = 3

        if FOV is None:
            FOV = self.image_size * 1.25

        print("Cutting out to a FOV of ", FOV, "arcsec")
        print("Velocity channels between", self.iv_min-iv_buffer, "and", self.iv_max+iv_buffer)
        print("ie, between", self.cube.velocity[self.iv_min-iv_buffer], "and", self.cube.velocity[self.iv_max+iv_buffer], "km/s")

        self.cube.cutout(new_file,FOV=FOV,overwrite=True,iv_min=self.iv_min-iv_buffer,iv_max=self.iv_max+iv_buffer)

        self.cutout_parameters = [FOV,self.iv_min-iv_buffer,self.iv_max+iv_buffer]

        return


    def _initial_guess(self,num=0,std=None):
        """
        Calculates standard deviation, PA, Dec, systemic velocity and stellar position. Does not override any parameters
        if they were provided.
        """

        #----------------------------
        # 1. Noise properties
        #----------------------------

        # Measure the standard deviation in 1st and last channels
        self.cube.get_std()
        std = self.cube.std
        print("Estimated std per channel is : ", std, self.cube.unit)

        # Image cube with no NaN
        if np.isnan(np.max(self.cube.image)):
            print("Forcing NaNs to 0")
            self.cube.image = np.nan_to_num(self.cube.image[:,:,:])
        image = self.cube.image


        #----------------------------
        # 2. Line profile analysis
        #----------------------------

        # Find the 1st and last channels with significant signal
        nv = self.cube.nv
        iv_min = nv-1
        iv_max = 0
        min_image = np.min(image,axis=0) # proxy for continuum if present
        for i in range(nv):
            if np.max(image[i,:,:] - min_image) > 10*std:
                iv_min = np.minimum(iv_min,i)
                iv_max = np.maximum(iv_max,i)

        self.iv_min = iv_min
        self.iv_max = iv_max

        print("Signal detected over channels:", iv_min, "to", iv_max)
        print("i.e. velocity range:", self.cube.velocity[[iv_min, iv_max]], "km/s")

        # Extracting the 2 brightest channels and estimate v_syst
        profile = self.cube.get_line_profile(threshold=3*self.cube.std)

        profile_rms = np.std(profile[:np.maximum(iv_min,10)])

        dv = np.abs(self.cube.velocity[1]-self.cube.velocity[0])
        # We have at least 0.5km/s between peaks
        dx = np.maximum(4,int(0.25/dv))

        iv_peaks = search_maxima(profile, height=10*profile_rms, dx=dx, prominence=0.05*np.max(profile))

        plt.figure(num+1)
        plt.clf()
        self.cube.plot_line(threshold=3*std)

        if (iv_peaks.size > 2):
            print("*** WARNING: Found more than 2 peaks in the line profile : please double check estimated values")
        #plt.plot(self.cube.velocity[iv_peaks], profile[iv_peaks], "o")
        if (iv_peaks.size < 2):
            print("Could not find double peaked line profile, fitting a single peak")
            print("Brightest channel is :", iv_peaks[0])
            print("Velocity of brightest channels:", self.cube.velocity[iv_peaks[0]], "km/s")

            # Refine peak position by fitting a Gaussian
            x = self.cube.velocity[iv_min:iv_max]
            y = profile[iv_min:iv_max]

            p0 = [0.5*np.max(y), 0.5*np.max(y), self.cube.velocity[iv_peaks[0]], np.minimum(self.cube.velocity[iv_max] - self.cube.velocity[iv_min],0.2)]
            #plt.plot(x,Gaussian_p_cst(x,p0[0],p0[1],p0[2],p0[3]), color="red")

            p, _ = curve_fit(Gaussian_p_cst,x,y,sigma=1/np.sqrt(y), p0=p0)
            v_peaks = p[2]

            x2=np.linspace(np.min(x),np.max(x),100)
            plt.plot(x2,Gaussian_p_cst(x2,p[0],p[1],p[2],p[3]), color="C3", linestyle="--", lw=1)
            print("Velocity of peaks:", v_peaks, "km/s")

            self.delta_v_peaks = 0.0
        else:
            # We take the 2 brightest peaks, and we sort them in velocity
            iv_peaks = iv_peaks[:2]
            iv_peaks = iv_peaks[np.argsort(self.cube.velocity[iv_peaks])]
            print("Brightest channels are :", iv_peaks[:2])
            print("Velocity of brightest channels:", self.cube.velocity[iv_peaks[:2]], "km/s")

            # Refine peak position by fitting a Gaussian
            d_iv = np.ceil(0.25 * (iv_peaks[1]-iv_peaks[0])).astype(int)
            v_peaks=np.zeros(2)
            for i in range(2):
                x = self.cube.velocity[iv_peaks[i]-d_iv:iv_peaks[i]+d_iv+1]
                y = profile[iv_peaks[i]-d_iv:iv_peaks[i]+d_iv+1]

                p0 = [0.5*np.max(y), 0.5*np.max(y), self.cube.velocity[iv_peaks[i]], np.minimum(self.cube.velocity[iv_peaks[1]] - self.cube.velocity[iv_peaks[0]],0.2)]
                #plt.plot(x,Gaussian_p_cst(x,p0[0],p0[1],p0[2],p0[3]), color="red")

                p, _ = curve_fit(Gaussian_p_cst,x,y,sigma=1/np.sqrt(y), p0=p0)
                v_peaks[i] = p[2]

                x2=np.linspace(np.min(x),np.max(x),100)
                plt.plot(x2,Gaussian_p_cst(x2,p[0],p[1],p[2],p[3]), color="C3", linestyle="--", lw=1)

            self.delta_v_peaks = v_peaks[1] - v_peaks[0]
            print("Velocity of peaks:", v_peaks, "km/s")

        plt.xlabel("Velocity (km/s)")
        plt.ylabel("Flux density (Jy/beam)")

        self.iv_peaks = iv_peaks
        self.v_peaks = v_peaks

        self.v_syst = np.mean(self.v_peaks)
        print("Estimated systemic velocity (peaks) =", self.v_syst, "km/s")

        plt.plot([self.v_syst,self.v_syst], [0.,1.05*np.max(profile)], lw=1, color="C3", alpha=0.7)

        # --------------------------------
        # Fitting v_syst from line wings
        # --------------------------------
        background = 0.5 * (profile[iv_min]+profile[iv_max])
        delta_profile = np.max(profile) - background

        ou = np.array(np.where(profile < background + 0.05 * delta_profile))
        ou = ou[(ou > iv_min) * (ou < iv_max)]

        x = self.cube.velocity[ou]
        y = profile[ou]

        p0 = [0, 2*p[1], self.v_syst, 0.5*(self.cube.velocity[iv_max] - self.cube.velocity[iv_min])]
        #plt.plot(x,Gaussian_p_cst(x,p0[0],p0[1],p0[2],p0[3]), color="red")

        p, _ = curve_fit(Gaussian_p_cst,x,y,sigma=1/np.sqrt(y), p0=p0)
        v_syst_wings = p[2]
        self.v_syst_wings = v_syst_wings
        print("Estimated systemic velocity (wings) =", v_syst_wings, "km/s")

        ou1 = ou[x < v_syst_wings]
        plt.plot(self.cube.velocity[ou1], profile[ou1], color="green")
        ou2 = ou[x > v_syst_wings]
        plt.plot(self.cube.velocity[ou2], profile[ou2], color="green")

        x2=np.linspace(np.min(x),np.max(x),100)
        plt.plot(x2,Gaussian_p_cst(x2,p[0],p[1],p[2],p[3]), color="green", linestyle="--", lw=1)

        plt.plot([v_syst_wings,v_syst_wings], [0.,1.05*np.max(profile)], lw=1, color="green", alpha=0.7)
        #----

        self.excluded_delta_v = np.maximum(0.25 * self.delta_v_peaks, 0.1)
        print("Excluding channels within ", self.excluded_delta_v, "km/s of systemic velocity" )

        iv_syst = np.argmin(np.abs(self.cube.velocity - self.v_syst))
        self.iv_syst = iv_syst
        iv_syst_wings = np.argmin(np.abs(self.cube.velocity - self.v_syst_wings))
        self.iv_syst_wings = iv_syst_wings
        print("Closest channel to systemic is", iv_syst)
        print("Offset from systemic is", self.cube.velocity[iv_syst] - self.v_syst,"km/s ==",
              np.abs((self.cube.velocity[iv_syst] - self.v_syst)/(self.cube.velocity[1]-self.cube.velocity[0])), "channel width")

        #---------------------------------
        # 3. Estimated stellar position
        #---------------------------------
        # We want at least 15 sigma to detect the star
        iv_min = nv
        iv_max = 0
        for i in range(self.iv_min,self.iv_max):
            if np.max(image[i,:,:]) > 15*std:
                iv_min = np.minimum(iv_min,i)
                iv_max = np.maximum(iv_max,i)

        dv = np.abs(np.minimum(iv_syst_wings-iv_min, iv_max-iv_syst_wings) - 1)

        iv1 = np.maximum(int(iv_syst_wings-dv),0)
        iv2 = np.minimum(int(iv_syst_wings+dv),nv-1)

        plt.figure(num+1)
        print(iv1, iv2 )
        plt.plot(self.cube.velocity[[iv1,iv2]], profile[[iv1,iv2]], "o")
        plt.xlim(self.cube.velocity[iv1]-0.5,self.cube.velocity[iv2]+0.5)

        x = np.zeros(2)
        y = np.zeros(2)
        for i, iv in enumerate([iv1,iv2]):
            im = image[iv,:,:]
            im = np.where(im > 7*std, im, 0)
            c = ndimage.center_of_mass(im)
            x[i] = c[1]
            y[i] = c[0]

        PA_inner = np.rad2deg(np.arctan2(y[1]-y[0],x[1]-x[0])) - 90
        self.PA_inner_disc = PA_inner
        print("Estimated PA of red shifted side (inner disc) =", PA_inner, "deg")

        x_star = np.mean(x)
        y_star = np.mean(y)
        self.x_star = x_star
        self.y_star = y_star
        print("Finding star. Using channels", iv1, iv2)
        print("Estimated position of star:", x_star, y_star, "(pixels)")

        self.dRA = ((self.cube.nx-1)/2 - x_star) * self.cube.pixelscale
        self.dDec = (y_star - (self.cube.ny-1)/2) * self.cube.pixelscale
        print(f'dRA={self.dRA:<5.3f}" dDec={self.dDec:<5.3f}"')

        color=["C0","C3"] # ie blue and red
        if (plt.fignum_exists(num+2)):
            plt.figure(num+2)
            plt.clf()
        fig2, axes2 = plt.subplots(nrows=1,ncols=2,num=num+2,figsize=(12,5),sharex=True,sharey=True)
        for i in range(2):
            ax = axes2[i]
            self.cube.plot(iv=[iv1,iv2][i],ax=ax,axes_unit="pixel")
            ax.plot(x[i],y[i],"o",color=color[i],ms=2)
            ax.plot(self.x_star,self.y_star,"*",color="white",ms=3)

        #---------------------------------
        # 4. Estimated disk orientation
        #---------------------------------

        # Measure centroid in 2 brightest channels
        x = np.zeros(2)
        y = np.zeros(2)
        if len(iv_peaks) == 2:
            iv_channels = iv_peaks
        else:
            # if we have a single peak line profile, we use the channels half-way
            dv = (np.minimum(iv_syst-iv_min, iv_max-iv_syst) - 1)//2
            iv_channels = [int(iv_syst-dv),int(iv_syst+dv)]

            print("test", iv_channels)

        for i, iv in enumerate(iv_channels):
            im = image[iv,:,:]
            im = np.where(im > 3*std, im, 0)
            c = ndimage.center_of_mass(im)
            x[i] = c[1]
            y[i] = c[0]

        PA = np.rad2deg(np.arctan2(y[1]-y[0],x[1]-x[0])) - 90
        self.PA = PA
        print("Estimated PA of red shifted side =", PA, "deg")

        if plt.fignum_exists(num + 3):
            plt.figure(num+3)
            plt.clf()
        fig3, axes3 = plt.subplots(nrows=1,ncols=2,num=num+3,figsize=(12,5),sharex=True,sharey=True)
        for i in range(2):
            ax = axes3[i]
            self.cube.plot(iv=iv_channels[i],ax=ax,axes_unit="pixel")
            ax.plot(x[i],y[i],"o",color=color[i],ms=4)
            ax.plot(x[i],y[i],"o",color="white",ms=2)

            ax.plot(self.x_star,self.y_star,"*",color="white",ms=3)

        # Measure sign of inclination from average of 2 centroid, using a cross product with red shifted side
        # positive inclination means that the near side of the upper surface is at the bottom of the map when the
        # blue-shifted side is to the right
        # this is the opposite convention in mcfost !!!!
        self.is_inc_positive = (np.mean(x)-x_star)*(y[1]-y_star) - (np.mean(y)-y_star)*(x[1]-x_star) > 0.

        if self.is_inc_positive:
            print("Inclination angle is positive")
        else:
            print("Inclination angle is negative")

        if self.is_inc_positive:
            self.inc_sign = 1
        else:
            self.inc_sign = -1

        #---------------------------------
        # 4. Estimated size of image
        #---------------------------------
        self._get_image_size(num=num)

        return

    def _rotate_cube(self):

        # Rotate star position
        angle = np.deg2rad(self.PA - self.inc_sign * 90.0)
        center = (np.array(self.cube.image.shape[1:3])-1)/2.
        dx = self.x_star-center[0]
        dy = self.y_star-center[1]
        self.x_star_rot = center[0] + dx * np.cos(angle) + dy * np.sin(angle)
        self.y_star_rot = center[1] - dx * np.sin(angle) + dy * np.cos(angle)

        with alive_bar(int(self.iv_max-self.iv_min), title="Rotating cube") as bar:
            for iv in range(self.iv_min,self.iv_max):
                self.cube.image[iv,:,:] = np.array(rotate(self.cube.image[iv,:,:], self.PA - self.inc_sign * 90.0, reshape=False))
                bar()

        return

    def _get_image_size(self,num=0):
        # get size of image with flux
        plt.figure(num+7)
        plt.clf()
        self.cube.plot(moment=0, threshold=3*self.cube.std, iv_support=np.arange(self.iv_min,self.iv_max+1),axes_unit="pixel")
        M0 = self.cube.last_image

        # Find x index of pixels with signals
        ou = np.where(np.isfinite(np.nanmax(M0,axis=0)))
        disk_size = (np.max(ou) - np.min(ou))  * self.cube.pixelscale
        self.disk_size = disk_size

        # Find y index of pixels with signals
        ou = np.where(np.isfinite(np.nanmax(M0,axis=1)))
        disk_size2 = (np.max(ou) - np.min(ou))  * self.cube.pixelscale

        self.image_size = np.maximum(disk_size,disk_size2)

        print("Actual size of image (ie with flux) is ~", self.image_size, "arcsec")

        return


    def _get_disk_size(self,num=0):
        plt.figure(num+8)
        plt.clf()
        self.cube.plot(moment=0, threshold=5*self.cube.std, iv_support=np.arange(self.iv_min,self.iv_max+1),axes_unit="pixel")
        M0 = self.cube.last_image

        # We have rotated the cube beforehand
        self.M0 = M0

        # Find x index of pixels with signals
        ou = np.where(np.isfinite(np.nanmax(M0,axis=0)))
        disk_size = (np.max(ou) - np.min(ou))  * self.cube.pixelscale
        self.disk_size = disk_size

        # Find y index of pixels with signals
        ou = np.where(np.isfinite(np.nanmax(M0,axis=1)))
        disk_size2 = (np.max(ou) - np.min(ou))  * self.cube.pixelscale

        print("Disk size is ", disk_size, "x", disk_size2, " arcsec")
        print("Aspect ratio suggests inclination close to:", np.arccos(disk_size2/disk_size) * 180./np.pi), " deg"

        self.disk_size = [disk_size,disk_size2]

        return

    def _select_scales(self,num=0):
        # Estimating the taper to use for the multi-scale analysis
        #  2**n/2 * bmin up to a fraction of the disk size along semi-major axis

        n_beams = self.disk_size/self.cube.bmin

        print("There are ", n_beams, " beams accros the disk semi-major axis")

        # We want at least 5 beams per side of the disk
        n_scales = int(np.ceil(np.log2(n_beams/3.))) # Number of scales with a factor 2

        n_scales = 2*n_scales-1 # Number of scales with a factor sqrt(2)
        f = np.sqrt(2.)

        self.n_scales = n_scales
        self.scales = self.cube.bmin * f**(np.arange(n_scales))

        return

    def _create_rotated_cube(self):

        self.rotated_images = np.zeros((self.n_scales,self.iv_max-self.iv_min+1,self.cube.ny,self.cube.nx), dtype=np.float32)
        self.rotated_images[0,:,:,:] = self.cube.image[self.iv_min:self.iv_max+1,:,:] # scale 0

        return


    def _extract_isovelocity(self):
        """
        Infer the upper emission surface from the provided cube
        extract the emission surface in each channel and loop over channels

        Args:
        cube (casa instance): An imgcube instance of the line data.

        PA (float): Position angle of the source in [degrees].
        y_star (optional) : position of star in  pixel (in rotated image), used to filter some bad data
        without y_star, more points might be rejected
        """

        cube = self.cube
        nx, nv = cube.nx, cube.nv
        ns = self.n_scales

        # todo : add scales
        self.n_surf = np.zeros([ns,nv], dtype=int)
        self.x_sky = np.zeros([ns,nv,nx])
        self.y_sky = np.zeros([ns,nv,nx,2])
        self.Tb = np.zeros([ns,nv,nx,2])
        self.I = np.zeros([ns,nv,nx,2])

        # Loop over the channels
        with alive_bar(int(self.iv_max-self.iv_min), title="Extracting isovelocity curves") as bar:
            for iv in range(self.iv_min,self.iv_max):
                for iscale in range(self.n_scales):
                    self._extract_isovelocity_1channel(iv,iscale)
                    #self._refine_isovelocity_1channel(iv,iscale=iscale)
                    bar()

        #--  Additional spectral filtering to clean the data ??


        ou = np.where(self.n_surf[0,:]>1)
        self.iv_min_surf = np.min(ou)
        self.iv_max_surf = np.max(ou)
        print("Surfaces detected between channels", self.iv_min_surf, "and", self.iv_max_surf)

        self.snr = self.I/self.cube.std

        return


    def _make_multiscale_cube(self):

        self.multiscale_bmaj=np.zeros(self.n_scales)
        self.multiscale_bmin=np.zeros(self.n_scales)
        self.multiscale_std=np.zeros(self.n_scales)

        self.multiscale_bmaj[0] = self.cube.bmaj
        self.multiscale_bmin[0] = self.cube.bmin

        self.multiscale_std[0] = self.cube.std

        # Loop over scales
        for iscale in range(self.n_scales):

            if iscale > 0:
                taper = self.scales[iscale]
                # --- Convolution : note : todo: we only want to rotate once as this slow
                if taper < self.cube.bmaj:
                    delta_bmaj = self.cube.pixelscale * FWHM_to_sigma # sigma will be 1 pixel
                    bmaj = self.cube.bmaj + delta_bmaj
                else:
                    delta_bmaj = np.sqrt(taper ** 2 - self.cube.bmaj ** 2)
                    bmaj = taper
                delta_bmin = np.sqrt(taper ** 2 - self.cube.bmin ** 2)
                bmin = taper

                sigma_x = delta_bmin / self.cube.pixelscale * FWHM_to_sigma  # in pixels
                sigma_y = delta_bmaj / self.cube.pixelscale * FWHM_to_sigma  # in pixels

                beam = Gaussian2DKernel(sigma_x, sigma_y, self.cube.bpa * np.pi / 180)

                # Measure new std
                #self.multiscale_std[iscale] = self.cube.std * (self.cube.bmaj * self.cube.bmin) / (bmaj * bmin) # this underestimayte std at at high scales
                im = convolve_fft(self.cube.image[0,:,:], beam)
                im1 = convolve_fft(self.cube.image[-1,:,:], beam)
                self.multiscale_std[iscale] = np.nanstd([im,im1])

                # Make the multiscale cube
                with alive_bar(int(self.iv_max-self.iv_min), title="Making multi-scale cube: scale #"+str(iscale)) as bar:
                    for iv in range(self.iv_min,self.iv_max):
                        im = self.rotated_images[0,iv-self.iv_min,:,:]
                        self.rotated_images[iscale,iv-self.iv_min,:,:] = convolve_fft(im, beam)
                        bar()

                self.multiscale_bmaj[iscale] = bmaj
                self.multiscale_bmin[iscale] = bmin

        return


    def _extract_isovelocity_1channel(self,iv,iscale=0):
        # Find the maxima along y as in Pinte at al. 2018


        clean_method1 = True # removing points where the upper surface or average surface is below star at a given x
        clean_method2 = True # removing points that deviate a lot as a function of x
        clean_method3 = False # Himanshi's volatility

        quadratic_fit = True # refine position with a quadratic fit

        if np.abs(self.cube.velocity[iv] - self.v_syst) < self.excluded_delta_v:
            self.n_surf[iscale,iv] = 0
            return

        nx = self.cube.nx

        im = self.rotated_images[iscale,iv-self.iv_min,:,:]
        std = self.multiscale_std[iscale]
        bmaj = self.multiscale_bmaj[iscale]

        # Setting up arrays in each channel map
        in_surface = np.full(nx,False)
        j_surf = np.zeros([nx,2], dtype=int)
        j_surf_exact = np.zeros([nx,2])
        T_surf = np.zeros([nx,2])
        I_surf = np.zeros([nx,2])

        # Selecting range of pixels to explore depending on velocity (ie 1 side of the disk only)
        if (self.cube.velocity[iv] - self.v_syst) * self.inc_sign > 0:
            i1=0
            i2=int(np.floor(self.x_star_rot))-1
        else:
            i1=int(np.floor(self.x_star_rot))+2
            i2=nx

        # Loop over the pixels along the x-axis to find surface
        for i in range(i1,i2):
            vert_profile = im[:,i]
            # find the maxima in each vertical cut, at signal above X sigma
            # ignore maxima not separated by at least a beam
            # maxima are ordered by decarasing flux
            j_max = search_maxima(vert_profile, height=self.sigma*std, dx=bmaj/self.cube.pixelscale,
                                  prominence=2*std)

            if j_max.size>1:  # We need at least 2 maxima to locate the surface
                in_surface[i] = True

                # indices of the near [0] and far [1] sides of upper surface
                j_surf[i,:] = np.sort(j_max[:2])

                # exclude maxima that do not make sense : only works if upper surface is at the top
                if clean_method1:
                    if j_surf[i,1] - self.y_star_rot < 0:
                        #print("pb 1 iv=", iv, "i=", i, "j=", j_surf[i,1])
                        # Houston, we have a pb : the far side of the disk cannot appear below the star
                        j_max_sup = j_max[np.where(j_max > self.y_star_rot)]
                        if j_max_sup.size:
                            j_surf[i,1] = j_max_sup[0]
                            j_surf[i,0] = j_max[0]
                        else:
                            in_surface[i] = False

                    if np.mean(j_surf[i,:]) - self.y_star_rot < 0:
                        #print("pb 2 iv=", iv, "i=", i, "j=", j_surf[i,:])
                        # the average of the top surfaces cannot be below the star
                        in_surface[i] = False

                #-- We find a spatial quadratic to refine position of maxima (like bettermoment does in velocity)
                if quadratic_fit:
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
                        I_surf[i,k] = im[j,i] = f_max
                else:
                    for k in range(2):
                        j = j_surf[i,k]
                        j_surf_exact[i,k] = j
                        I_surf[i,k] = im[j,i]

                T_surf[i,k] = self.cube._Jybeam_to_Tb(I_surf[i,k]) # Converting to Tb (currently assuming the cube is in Jy/beam)

        #-- Now we try to clean out a bit the surfaces we have extracted

        #-- We test if front side is too high or the back side too low
        # this happens when the data gets noisy or diffuse and there are local maxima
        # fit a line to average curve and remove points from front if above average
        # and from back surface if  below average (most of these case should have been dealt with with test on star position)
        # could search for other maxima but then it means that data is noisy anyway
        #e.g. measure_surface(HD163, 45, plot=True, PA=-45,plot_cut=503,sigma=10, y_star=478)

        x = np.arange(nx)

        if clean_method2:
            if np.any(in_surface):
                x1 = x[in_surface]

                y1 = np.mean(j_surf_exact[in_surface,:],axis=1)

                if len(x1) > 2:
                    P = np.polyfit(x1,y1,1)

                    # x_plot = np.array([0,nx])
                    # plt.plot(x_plot, P[1] + P[0]*x_plot)

                    #in_surface_tmp = in_surface &  (j_surf_exact[:,0] < (P[1] + P[0]*x)) # test only front surface
                    in_surface_tmp = in_surface & (j_surf_exact[:,0] < (P[1] + P[0]*x)) & (j_surf_exact[:,1] > (P[1] + P[0]*x))

                    # We remove the weird point and reddo the fit again to ensure the slope we use is not too bad
                    x1 = x[in_surface_tmp]
                    y1 = np.mean(j_surf_exact[in_surface_tmp,:],axis=1)
                    P = np.polyfit(x1,y1,1)

                    #in_surface = in_surface &  (j_surf_exact[:,0] < (P[1] + P[0]*x)) # test only front surface
                    in_surface = in_surface & (j_surf_exact[:,0] < (P[1] + P[0]*x)) & (j_surf_exact[:,1] > (P[1] + P[0]*x))

                    #-- test if we have points on both side of the star
                    # - remove side with the less points

        # Saving the data
        n = np.sum(in_surface) # number of points in that surface
        self.n_surf[iscale,iv] = n
        if n:
            self.x_sky[iscale,iv,:n] = x[in_surface]
            self.y_sky[iscale,iv,:n,:] = j_surf_exact[in_surface,:]
            self.Tb[iscale,iv,:n,:] = T_surf[in_surface,:]
            self.I[iscale,iv,:n,:] = I_surf[in_surface,:]

        return


    def _refine_isovelocity_1channel(self,iv,iscale=0):
        # Find iteratively the maxima along the perpendicular to the isovelocity curve
        # then remap on the initial regular x_spacing so we can compute the altitude

        # First we only keep points with no jump in x

        n = self.n_surf[iscale,iv]
        x = self.x_sky[iscale,iv,:n]
        y = self.y_sky[iscale,iv,:n,1] # 1 surface for now

        nbeams = 2. # cut along 1 beam ??
        f =  nbeams*self.multiscale_bmaj[iscale] / self.cube.pixelscale
        npix = int(np.floor(f)) # number of "pixels" along cut

        x_new = np.zeros(n)
        y_new = np.zeros(n)

        # We smooth a bit the previous extraction to avoid too much randomess in the cut direction
        y = signal.savgol_filter(y, window_length=5, polyorder=3, mode="nearest")

        for i in range(n):
            xc = x[i]
            yc = y[i]

            j = np.maximum(np.minimum(i,n-2),1) # dealing with ends
            dy = -(x[j+1]-x[j-1])
            dx = y[j+1]-y[j-1]

            norm = np.sqrt(dx**2 + dy**2)
            dy /= norm
            dx /= norm

            # We flip sign of cut if dx is <0
            if dx<0:
                dx=-dx
                dy=-dy

            x0 = xc - f * dx
            y0 = yc - f * dy

            x1 = xc + f * dx
            y1 = yc + f * dy

            if (i==30):
                print("----------------")
                print("xc=", xc, i, j)
                print("x=",  x[j+1], x[j-1])
                print("y=",  y[j+1], y[j-1])

                print("cut vector dxy=",dx,dy)

                print(x0, y0)
                print(x1,y1)

            x_cut, y_cut, z_cut = self.cube.make_cut(x0,y0,x1,y1, z = self.rotated_images[iscale,iv-self.iv_min,:,:], num=npix)

            imax = np.argmax(z_cut)

            x_new[i] = x_cut[imax]
            y_new[i] = y_cut[imax]

            if (i==30):
                print("new=", x_new[i], y_new[i])

        # reinterpolating new extracted surface on regular x
        f = interp1d(x_new,y_new,fill_value="extrapolate")
        y = f(x)

        self.x_sky[iscale,iv,:n] = x
        self.y_sky[iscale,iv,:n,1] = y


    def _compute_surface(self):
        """
        Deproject the detected surfaces to estimate, r,h, and v of the emitting layer

        inc (float): Inclination of the source in [degrees].
        """

        ### some things to note:
        # - PA : the code assumes that the disc near side must be in the bottom half of the image

        # Impact of parameters :
        # - value of x_star plays on dispersion of h and v
        # - value of PA creates a huge dispersion
        # - value of y_star shift the curve h(r) vertically, massive change on flaring exponent
        # - value of inc changes the shape a bit, but increases dispersion on velocity

        inc_rad = np.radians(self.inc)



        #-- Computing the radius and height for each point
        y_f = self.y_sky[:,:,:,1] - self.y_star_rot   # far side, y[channel number, x index]
        y_n = self.y_sky[:,:,:,0] - self.y_star_rot   # near side
        y_c = 0.5 * (y_f + y_n)
        #y_c = np.ma.masked_array(y_c,mask).compressed()

        x = self.x_sky[:,:,:] - self.x_star_rot

        # inclination plays a role from here
        y = (y_f - y_c) / np.cos(inc_rad)

        r = np.hypot(x,y) # Note : does not depend on y_star
        h = y_c / np.sin(inc_rad)

        v = (self.cube.velocity[np.newaxis,:,np.newaxis] - self.v_syst) * r / (x * np.sin(inc_rad)) # does not depend on y_star
        dv = (self.cube.velocity[np.newaxis,:,np.newaxis] - self.v_syst) * (r/r)

        r *= self.cube.pixelscale
        h *= self.cube.pixelscale

        # We eliminate the point where there is no detection
        mask = self.x_sky < 1

        mask = mask | np.isinf(v) | np.isnan(v)

        # -- we remove channels that are too close to the systemic velocity
        mask = mask | (np.abs(self.cube.velocity - self.v_syst) < self.excluded_delta_v)[:,np.newaxis]

        # -- we remove traces at small separation, if requested
        if self.exclude_inner_beam:
            mask = mask | (r < self.cube.bmaj)

        # -- If the disc is oriented the other way
        if np.median(h[~mask]) < 0:
            h = -h

        # -- we can now remove the points with h<0 (they correspond to values set to 0 in y)
        # and where v is not defined
        mask = mask | (h<0)

        r = np.ma.masked_array(r,mask)
        h = np.ma.masked_array(h,mask)
        v = np.ma.masked_array(v,mask)
        dv = np.ma.masked_array(dv,mask)

        # -- If the disc rotates in the opposite direction as expected
        if np.mean(v) < 0:
            v = -v

        # -- Todo : optimize position, inclination (is that posible without a model ?), PA (need to re-run detect surface)
        self.x = x
        self.y = y
        self.r = r
        self.h = h
        self.v = v
        self.dv = dv

        return

    def compute_v_std(self,nbins=30):

        r = self.r.compressed()
        h = self.h.compressed()
        v = self.v.compressed()
        T = np.mean(self.Tb[:,:,:,:],axis=3).ravel()[np.invert(self.r.mask.ravel())]

        h_std, _, _ = binned_statistic(r,h, 'std', bins=nbins)
        v_std, _, _ = binned_statistic(r,v, 'std', bins=nbins)
        T_std, _, _ = binned_statistic(r,T, 'std', bins=nbins)

        self.h_std = np.mean(h_std)
        self.v_std = np.mean(v_std)
        self.T_std = np.mean(T_std)
        self.v_std = self.v_std

        #self.v_std = np.mean(v_std) * np.mean(h_std) * np.mean(T_std)

        return

    def find_i(self,num=0):

        # Altitude dispersion
        plt.figure(num+5)
        plt.clf()

        # simple uniform gitd fit
        inc_array = np.arange(10,80,1)
        metric = np.zeros(len(inc_array))
        for i, inc in enumerate(inc_array):
            self.inc = inc
            self._compute_surface()
            self.compute_v_std()
            metric[i] = self.h_std

        plt.plot(inc_array,metric, color="red", markersize=1)
        plt.xlabel("Inclination ($^\mathrm{o}$)")
        plt.ylabel("altitude dispersion")

        # T dispersion
        plt.figure(num+6)
        plt.clf()

        # simple uniform gitd fit
        inc_array = np.arange(10,80,1)
        metric = np.zeros(len(inc_array))
        for i, inc in enumerate(inc_array):
            self.inc = inc
            self._compute_surface()
            self.compute_v_std()
            metric[i] = self.T_std

        plt.plot(inc_array,metric, color="red", markersize=1)
        plt.xlabel("Inclination ($^\mathrm{o}$)")
        plt.ylabel("T dispersion")

        # Velocity dispersion
        plt.figure(num+4)
        plt.clf()

        # simple uniform gitd fit
        inc_array = np.arange(10,80,1)
        metric = np.zeros(len(inc_array))
        for i, inc in enumerate(inc_array):
            self.inc = inc
            self._compute_surface()
            self.compute_v_std()
            metric[i] = self.v_std

        plt.plot(inc_array,metric, color="red", markersize=1)

        # We refine the fit
        inc = inc_array[np.nanargmin(metric)]
        inc_array = np.arange(inc-5,inc+5,0.25)
        metric = np.zeros(len(inc_array))
        for i, inc in enumerate(inc_array):
            self.inc = inc
            self._compute_surface()
            self.compute_v_std()
            metric[i] = self.v_std

        plt.plot(inc_array,metric, color="blue", markersize=1)
        plt.xlabel("Inclination ($^\mathrm{o}$)")
        plt.ylabel("Velocity dispersion")
        self.inc = inc_array[np.nanargmin(metric)]
        print("Best fit for inclination =", self.inc, "deg")

        return


    def plot_surfaces(self,
                      nbins: int = 30,
                      v_bin_width: float = None,
                      m_star: float = None,
                      m_star_h_func: float = None,
                      h_func: ndarray = None,
                      dist: float = None,
                      plot_power_law: bool = False,
                      plot_tapered_power_law: bool = False,
                      plot_gp: bool = False,
                      r0: float = 1.0,
                      save = None,
                      num = None,
                      scales = None
                      ):
        """
        Parameters
        ----------
        nbins
            Number of bins used to bin the data during plotting.
        v_bin_width
            If provided, this is the fixed radial bin width [arcsec] used for the binned velocity points only. Overrides
            nbins, though will have NaNs if a bin has no velocity points at that radial separation. exoALMA convention
            is 0.25 * beam semi-major axis.
        m_star
            Central mass for Keplerian rotation plot [Msun].
        m_star_h_func
            Central mass for Keplerian rotation plot based of a given height prescription [Msun].
        h_func
            Height prescription from a given fitted surface e.g. a power law fit.
        dist
            Distance to the source [pc]. Needed for saving the rotation curve in units of au (exoALMA convention)
        plot_power_law
            Fits the data with a power law and plots it.
        plot_tapered_power_law
            Fits the data with a tapered power law and plots it.
        r0
            Reference radius for surface height fits [arcsec]
        save
            Path as a string to save the plot.

        Returns
        -------
        CO_layers results figure.

        Notes
        -----
        Any notes?

        Other Parameters
        ----------------

        Examples
        --------

        """
        r = self.r[scales,:,:]
        h = self.h[scales,:,:]
        v = self.v[scales,:,:]
        dv = np.abs(self.dv[scales,:,:])
        T = np.mean(self.Tb[scales,:,:,:],axis=-1)

        r_data = r.ravel().compressed()#[np.invert(mask.ravel())]
        h_data = h.ravel().compressed()#[np.invert(mask.ravel())]
        v_data = v.ravel().compressed()#[np.invert(mask.ravel())]
        T_data = T.ravel()[np.invert(r.mask.ravel())]

        if plt.fignum_exists(num):
            plt.figure(num)
            plt.clf()
        fig = plt.figure(num,figsize=(15,5))
        gs = fig.add_gridspec(nrows=1,ncols=3)
        gs.update(wspace=0.2, hspace=0.05)
        ax=[]
        for i in range(0,1):
            for j in range(0,3):
                ax.append(fig.add_subplot(gs[i,j]))

        #Altitude
        ax[0].scatter(r.ravel(),h.ravel(),alpha=0.2,s=3, c=dv.ravel(), marker='o', label = 'data', cmap="jet")

        bins, _, _ = binned_statistic(r_data,[r_data,h_data], bins=nbins)
        std, _, _ = binned_statistic(r_data,h_data, 'std', bins=nbins)

        ax[0].errorbar(bins[0,:], bins[1,:],yerr=std, ecolor="grey", fmt='o', mec='k', mfc='grey', ms=3, elinewidth=2,
                       label='Binned data')



        if plot_power_law:
            #-- fitting a power-law
            P, C = self.fit_surface_height(r0 = r0)
            x = np.linspace(np.min(r),np.max(r),100)

            print('Power law fit: z0 = {:.5f} at {:.3f}", phi = {:.5f}'.format(P[0], r0, P[1]))

            ax[0].plot(x, P[0]*(x/r0)**P[1], color='k', ls='--', alpha=0.75, label='PL')

        if plot_tapered_power_law:
            #-- fitting a power-law
            P, C = self.fit_surface_height(tapered_power_law=True, r0=r0)
            x = np.linspace(np.min(r),np.max(r),100)

            print('Power law fit: z0 = {:.5f} at {:.3f}", phi = {:.5f}, r_taper = {:.5f}, q_taper = {:.5f}'
                  .format(P[0], r0, P[1], P[2], P[3]))

            ax[0].plot(x, P[0] * ((x/r0)**P[1]) * np.exp(-(x/P[2]) ** P[3]), color='k', ls='-.', alpha=0.75,
                       label='Tapered PL')

        if plot_gp:
            t, mu, std = self.fit_surface_height_gp()
            # Plot the data
            color = "#ff7f0e"
            ax[0].plot(t, mu, color="red", markersize=1.0)
            ax[0].fill_between(t, mu+std, mu-std, color=color, alpha=0.3, edgecolor="none")


        #Velocity
        ax[1].scatter(r.ravel(),v.ravel(),alpha=0.2,s=3,c=dv.ravel(),marker='o', label='Data',cmap="jet")

        if v_bin_width is not None:
            nbins = int(np.nanmax(r_data)/v_bin_width)  # rounds nbins down to the nearest int
            print('We used {} bins of width {:.4f} arcsec to bin the velocity data'.format(nbins, v_bin_width))

        bins, _, _ = binned_statistic(r_data, [r_data, v_data], bins=nbins)
        std, _, _ = binned_statistic(r_data, v_data, 'std', bins=nbins)

        # Generate a file with radius [au], velocity [km/s] and 1sigma dispersion [km/s] written as rows
        if dist is not None:
            filename = "{}_radius_vs_velocity.txt".format(self.cube.filename)
            np.savetxt(filename, (bins[0, :]*dist, bins[1, :], std))
            print("The radius [au], velocity [km/s] and 1\u03C3 dispersion [km/s] have been saved in "+filename)

        ax[1].errorbar(bins[0,:], bins[1,:],yerr=std, ecolor="grey", fmt='o', mec='k', mfc='grey', ms=3, elinewidth=2,
                       label='Binned data')

        if m_star:
            if dist is None:
                raise ValueError("dist must be provided")
            v_model = self._keplerian_disc(m_star, dist)
            ax[1].scatter(r_data, v_model,alpha=0.5,s=3,color="grey",marker='o', label = 'Kep model')

        if m_star_h_func:
            if dist is None:
                raise ValueError("dist must be provided")

            v_model = self._keplerian_disc(m_star_h_func, dist, h_func=h_func)

            x = np.sort(r_data)
            v_mod = -np.sort(-v_model)

            ax[1].plot(x, v_mod,alpha=0.75, ls='--',color="purple", label = 'Kep model w h_func')

        #Temperature
        sc = ax[2].scatter(r.ravel(),T.ravel(),alpha=0.5,s=3,c=dv.ravel(),marker='o',cmap="jet")
        colorbar2(sc)

        bins, _, _ = binned_statistic(r_data,[r_data,T_data], bins=nbins)
        std, _, _  = binned_statistic(r_data,T_data, 'std', bins=nbins)

        ax[2].errorbar(bins[0,:], bins[1,:],yerr=std, ecolor="grey", fmt='o', mec='k', mfc='grey', ms=3, elinewidth=2)

        ax[0].set_ylabel('Height ["]')
        ax[1].set_ylabel('Velocity [km/s]')
        ax[2].set_ylabel('Brightness Temperature [K]')

        ax[0].set_xlabel('Radius ["]')
        ax[1].set_xlabel('Radius ["]')
        ax[2].set_xlabel('Radius ["]')

        if save is not None and isinstance(save, str):
            plt.savefig(save, dpi=300, bbox_inches='tight', pad_inches=0.01, format='pdf')

        plt.show()

        return

    def plot_channel(self, iv, iscale=0, radius=3.0, ax=None, clear=True):

        if ax is None:
            ax = plt.gca()

        if clear:
            ax.cla()

        cube = self.cube
        x = self.x_sky
        y = self.y_sky
        n_surf = self.n_surf

        im = np.nan_to_num(self.rotated_images[iscale,iv-self.iv_min,:,:]) # array has been trimmed in velocity
        # Array is rotated already
        #if self.PA is not None:
        #    im = np.array(rotate(im, self.PA - self.inc_sign * 90.0, reshape=False))

        ax.imshow(im, origin="lower", cmap='binary_r')
        ax.set_title(r'v='+"{:.2f}".format(cube.velocity[iv])+' , $\Delta$v='+"{:.2f}".format(cube.velocity[iv] - self.v_syst)+' , id:'+str(iv), color='k')

        if n_surf[iscale,iv]:
            ax.plot(x[iscale,iv,:n_surf[iscale,iv]],y[iscale,iv,:n_surf[iscale,iv],0],"o",color="red",markersize=1)
            ax.plot(x[iscale,iv,:n_surf[iscale,iv]],y[iscale,iv,:n_surf[iscale,iv],1],"o",color="blue",markersize=1)
            ax.plot(x[iscale,iv,:n_surf[iscale,iv]],np.mean(y[iscale,iv,:n_surf[iscale,iv],:],axis=1),"o",color="white",markersize=1)

            # We zoom on the detected surfaces
            #ax.set_xlim(np.min(x[iv,:n_surf[iv]]) - 10*cube.bmaj/cube.pixelscale,np.max(x[iv,:n_surf[iv]]) + 10*cube.bmaj/cube.pixelscale)
            #ax.set_ylim(np.min(y[iv,:n_surf[iv],:]) - 10*cube.bmaj/cube.pixelscale,np.max(y[iv,:n_surf[iv],:]) + 10*cube.bmaj/cube.pixelscale)

        ax.plot(self.x_star_rot,self.y_star_rot,"*",color="yellow",ms=3)

        return


    def plot_channel_multiscale(self, iv, radius=3.0, ax=None, clear=True):
        # plot 1 channel and all the multiscale extraction

        if ax is None:
            ax = plt.gca()

        if clear:
            ax.cla()

        cube = self.cube
        x = self.x_sky
        y = self.y_sky
        n_surf = self.n_surf

        iscale=0
        im = np.nan_to_num(self.rotated_images[iscale,iv-self.iv_min,:,:])
        # Array is rotated already
        #if self.PA is not None:
        #    im = np.array(rotate(im, self.PA - self.inc_sign * 90.0, reshape=False))

        ax.imshow(im, origin="lower", cmap='binary_r')
        ax.set_title(r'$\Delta$v='+"{:.2f}".format(cube.velocity[iv] - self.v_syst)+' , id:'+str(iv), color='k')

        for iscale in range(self.n_scales):
            ax.plot(x[iscale,iv,:n_surf[iscale,iv]],y[iscale,iv,:n_surf[iscale,iv],0],"o",color="red",markersize=1)
            ax.plot(x[iscale,iv,:n_surf[iscale,iv]],y[iscale,iv,:n_surf[iscale,iv],1],"o",color="blue",markersize=1)
            ax.plot(x[iscale,iv,:n_surf[iscale,iv]],np.mean(y[iscale,iv,:n_surf[iscale,iv],:],axis=1),"o",color="white",markersize=1)

        return

    def plot_channel_multiscales(self, iv, radius=3.0, clear=True, num=21):
        # plot 1 channel and all the multiscale extraction on multiple panels

        ncols = int(np.ceil((self.n_scales+1)/2))
        fig, axs = plt.subplots(ncols=ncols, nrows=2, figsize=(2*ncols+1, 5),constrained_layout=True,num=num,
                                clear=False, sharex=True, sharey=True)
        cube = self.cube
        x = self.x_sky
        y = self.y_sky
        n_surf = self.n_surf

        iscale=0
        im = np.nan_to_num(self.rotated_images[iscale,iv-self.iv_min,:,:])
        # Array is rotated already
        #if self.PA is not None:
        #    im = np.array(rotate(im, self.PA - self.inc_sign * 90.0, reshape=False))

        ax0 = axs.ravel()[0]
        ax0.imshow(im, origin="lower", cmap='binary_r')
        ax0.set_title(r'$\Delta$v='+"{:.2f}".format(cube.velocity[iv] - self.v_syst)+' , id:'+str(iv), color='k')

        for iscale in range(self.n_scales):
            ax = axs.ravel()[iscale+1]

            im = np.nan_to_num(self.rotated_images[iscale,iv-self.iv_min,:,:])
            ax.imshow(im, origin="lower", cmap='binary_r')

            ax.plot(x[iscale,iv,:n_surf[iscale,iv]],y[iscale,iv,:n_surf[iscale,iv],0],"o",color="red",markersize=1)
            ax.plot(x[iscale,iv,:n_surf[iscale,iv]],y[iscale,iv,:n_surf[iscale,iv],1],"o",color="blue",markersize=1)
            ax.plot(x[iscale,iv,:n_surf[iscale,iv]],np.mean(y[iscale,iv,:n_surf[iscale,iv],:],axis=1),"o",color="white",markersize=1)

            ax0.plot(x[iscale,iv,:n_surf[iscale,iv]],y[iscale,iv,:n_surf[iscale,iv],0],"o",color="red",markersize=1)
            ax0.plot(x[iscale,iv,:n_surf[iscale,iv]],y[iscale,iv,:n_surf[iscale,iv],1],"o",color="blue",markersize=1)
            ax0.plot(x[iscale,iv,:n_surf[iscale,iv]],np.mean(y[iscale,iv,:n_surf[iscale,iv],:],axis=1),"o",color="white",markersize=1)

        return

    def plot_channels(self,n=20, num=21, radius=1.0, iv_min=None, iv_max=None, save=False, iscale=0):

        if iv_min is None:
            iv_min=self.iv_min_surf
        if iv_max is None:
            iv_max = self.iv_max_surf

        nv = iv_max-iv_min
        dv = nv/n

        ncols=5
        nrows = np.ceil(n / ncols).astype(int)

        if (plt.fignum_exists(num)):
            plt.figure(num)
            plt.clf()
        fig, axs = plt.subplots(ncols=5, nrows=nrows, figsize=(11, 2*nrows+1),constrained_layout=True,num=num,
                                clear=False)

        for i, ax in enumerate(axs.flatten()):
            iv = int(iv_min+i*dv)
            print(i, iv)
            self.plot_channel(iv, iscale=iscale, radius=radius, ax=ax)

        if save is not None and isinstance(save, str):
            plt.savefig(save, dpi=300, bbox_inches='tight', pad_inches=0.01, format='pdf')

        plt.show()

        return

    def fit_central_mass(self,
        initial_guess: float = 1.0,
        dist: float = None,
        h_func: ndarray = None,
        scales = None
        ):
        """
        Parameters
        ----------
        initial_guess
            initial guess of central mass [Msun].
        dist
            distance of source [pc].
        h_func
            Height prescription from a given fitted surface e.g. a power law fit.

            Setting this parameter will produce a m_star minimisation solution based
            off this height fit and not the scattered height data.

        Returns
        -------
        Minimised central mass solution.

        Notes
        -----
        Any notes?

        Other Parameters
        ----------------

        Examples
        --------

        """
        if dist is None:
            raise ValueError("dist must be provided")

        initial = np.array([initial_guess])

        soln = minimize(self._ln_like, initial, bounds=((0, None),), args=(dist, h_func))

        print("Fitting central mass, maximum likelihood estimate:")
        self.m_star_sol = soln
        self.m_star = soln.x[0]
        print("Central mass = {:.5f} Msun".format(self.m_star))

        return

    def _ln_like(self, theta, dist, h_func = None, scales=None):
        """Compute the ln like.
        """

        #define param values
        m_star = theta[0]

        # compute the model for the chi2
        if h_func is not None:
            v_model = self._keplerian_disc(m_star, dist, h_func=h_func)
        else:
            v_model = self._keplerian_disc(m_star, dist, h_func=h_func)

        v = self.v[scales,:,:].ravel().compressed()

        #using 1/snr as the error on the velocity
        v_error = 1/(np.mean(self.snr[scales,:,:,:],axis=-1).ravel()[np.invert(self.r[scales,:,:].mask.ravel())])

        # chi2
        chi2 = np.sum(((v - v_model)**2 / v_error**2) + np.log(2*np.pi*v_error**2))
        return 0.5 * chi2

    def _keplerian_disc(self, m_star, dist, h_func=None):
        """M_star for a keplerian disc"""
        #Defining constants
        G = sc.G
        msun = ac.M_sun.value

        r = self.r.ravel().compressed() * dist * sc.au

        if h_func is not None:
            h = h_func * dist * sc.au
        else:
            h = self.h.ravel().compressed() * dist * sc.au

        v = np.sqrt((G*m_star*msun*r**2)/((r**2 + h**2)**(3/2)))/1000
        return v

    def fit_surface_height(self,
        r0: float = 1.0,
        tapered_power_law: bool = False
        ):
        """
        Parameters
        ----------
        r0
            Reference radius for surface height fits.
        tapered_power_law
            Alternatively fit a tapered power law.

        Returns
        -------
        A functional fit to the emitting surface.

        Notes
        -----
        Any notes?

        Other Parameters
        ----------------

        Examples
        --------

        """

        r = np.array(self.r.ravel().compressed())
        h = self.h.ravel().compressed()
        error = 1/(np.mean(self.snr[:,:,:,:],axis=-1).ravel()[np.invert(self.r.mask.ravel())])

        if tapered_power_law:

            bnds = ((0.0, 0.0, r.min(), r.min()),(5.0, 5.0, 5.0, 2*r.max()))

            def func(r, z0, phi, r_taper, q_taper):
                return z0*((r/r0)**phi) * np.exp(-(r/r_taper)**q_taper)
        else:

            bnds = ((0.0,0.0),(5.0,5.0))

            def func(r, z0, phi):
                return z0*(r/r0)**phi

        popt, copt = curve_fit(func, r, h, sigma = error, bounds=bnds, maxfev = 100000)

        return popt, copt


    def to_mcfost(self, planet_r=0., planet_PA=0.):
        # For a given planet projected separation and PA,
        # this function gives the mcfost inclination, as well as the
        # az to be passed to the planet_az option and the deprojected separation
        #
        # input:
        # ------
        # dynamite model + planet radius in arcsec or au, and planet PA in deg (in plane of sky, East from North)
        #
        # output:
        # -------
        # inclination in degrees to set in parameter file
        # planet radius (eg to set in hydro simulations). It will be in same unit as planet_r
        # planet_az in degrees (this is passed to mcfost via cmd line option)

        # Note that pymcfost.get_planet_rPA does the opposite from a mcfost image

        inc = self.inc * self.inc_sign

        dPA = planet_PA - self.PA



        #az = np.arctan(np.tan(np.deg2rad(dPA)) / np.cos(np.deg2rad(self.inc)))

        y = np.sin(np.deg2rad(dPA)) / np.cos(np.deg2rad(inc))
        x = np.cos(np.deg2rad(dPA))
        az = np.arctan2(y,x)

        az = - np.rad2deg(az) # conversion to deg and correct convention for mcfost

        y_p = planet_r * np.sin(np.deg2rad(dPA))
        x_p = planet_r * np.cos(np.deg2rad(dPA))

        x = x_p
        y = y_p / np.cos(np.deg2rad(inc))

        r = np.hypot(x,y)

        # test :
        #mcfost.get_planet_r_az(62.5,50.2, 0.60521173 * 157.2, 11.619613647460938)
        # should give : (130.00000395126042, 62.500002877567475)

        # mcfost inclination is opposite to dynamite (which mattches discminer for convenience)
        if (inc<0):
            mcfost_inc=-inc
        else: # to avoid the bug in red/blue PA with negative inclination in mcfost
            mcfost_inc=180-inc
            az = -az

        print("MCFOST parameters should be:")
        print("i=",mcfost_inc,"deg")
        print("PA=",self.PA,"deg")
        print("planet r=",r,"au")
        print("planet az=",az,"deg (for cmd line option)")

        return mcfost_inc, r, az


    def fit_surface_height_gp(self):

        x = np.array(self.r.ravel().compressed())
        y = self.h.ravel().compressed()
        yerr = 1/(np.mean(self.snr[:,:,:,:],axis=3).ravel()[np.invert(self.r.mask.ravel())])

        order=np.argsort(x)
        x=x[order]
        y=y[order]
        yerr=yerr[order]

        # Set up the GP model
        k0 = terms.JitterTerm(log_sigma=np.log(np.var(y)))
        k1 = terms.RealTerm(log_a=np.log(np.var(y)), log_c=np.log(50))
        k2 = terms.RealTerm(log_a=np.log(np.var(y)), log_c=np.log(5))
        kernel = k0+k1+k2
        kernel = k0+k2
        gp = celerite.GP(kernel, fit_mean=False)

        # Define a cost function
        def neg_log_like(params, y, gp):
            gp.set_parameter_vector(params)
            return -gp.log_likelihood(y)

        def grad_neg_log_like(params, y, gp):
            gp.set_parameter_vector(params)
            return -gp.grad_log_likelihood(y)[1]

        gp.compute(x, yerr)

        initial_params = gp.get_parameter_vector()
        bounds = gp.get_parameter_bounds()

        #soln = minimize(nll, initial_params, jac=True)
        soln = minimize(neg_log_like, initial_params, jac=grad_neg_log_like,method="L-BFGS-B", bounds=bounds, args=(y, gp))
        gp.set_parameter_vector(soln.x)

        # Make the maximum likelihood prediction
        t = np.linspace(np.min(x), np.max(x), 500)

        mu, var = gp.predict(y, t, return_var=True)
        std = np.sqrt(var)

        return t, mu, std


def search_maxima_old(y, height=None, dx=0, prominence=0):
    """
    Returns the indices of the maxima of a function
    Indices are sorted by decreasing values of the maxima

    Args:
         y : array where to search for maxima
         threshold : minimum value of y for a maximum to be detected
         dx : minimum spacing between maxima [in pixel]


    Note : this is 50% faster than scipt but do not have promimence
    """

    # A maxima is a positive dy followed by a negative dy
    dy = y[1:] - y[:-1]
    i_max = np.where((np.hstack((0, dy)) > 0) & (np.hstack((dy, 0)) < 0))[0]

    if height:
        i_max = i_max[np.where(y[i_max]>height)]

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


def search_maxima(y, height=None, dx=0, prominence=0):
    """
    Returns the indices of the maxima of a function
    Indices are sorted by decreasing values of the maxima

    Args:
         y : array where to search for maxima
         threshold : minimum value of y for a maximum to be detected
         dx : minimum spacing between maxima [in pixel]
    """

     # find local maxima
    i_max, _ = find_peaks(y, distance = dx, width = 0.5*dx, height = height, prominence = prominence)

    # Sort the peaks by height
    i_max = i_max[np.argsort(y[i_max])][::-1]

    return i_max


def Gaussian_p_cst(x, C, A, x0, sigma):
    """" Gaussian + constant function """
    return C + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def colorbar2(mappable, shift=None, width=0.05, ax=None, trim_left=0, trim_right=0, side="right",**kwargs):
    # creates a color bar that does not shrink the main plot or panel
    # only works for horizontal bars so far

    if ax is None:
        ax = mappable.axes

    # Get current figure dimensions
    try:
        fig = ax.figure
        p = np.zeros([1,4])
        p[0,:] = ax.get_position().get_points().flatten()
    except:
        fig = ax[0].figure
        p = np.zeros([ax.size,4])
        for k, a in enumerate(ax):
            p[k,:] = a.get_position().get_points().flatten()
    xmin = np.amin(p[:,0]) ; xmax = np.amax(p[:,2]) ; dx = xmax - xmin
    ymin = np.amin(p[:,1]) ; ymax = np.amax(p[:,3]) ; dy = ymax - ymin

    if side=="top":
        if shift is None:
            shift = 0.2
        cax = fig.add_axes([xmin + trim_left, ymax + shift * dy, dx - trim_left - trim_right, width * dy])
        cax.xaxis.set_ticks_position('top')
        return fig.colorbar(mappable, cax=cax, orientation="horizontal",**kwargs)
    elif side=="right":
        if shift is None:
            shift = 0.05
        cax = fig.add_axes([xmax + shift*dx, ymin, width * dx, dy])
        cax.xaxis.set_ticks_position('top')
        return fig.colorbar(mappable, cax=cax, orientation="vertical",**kwargs)
