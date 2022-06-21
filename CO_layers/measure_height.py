from scipy.ndimage import rotate, shift
from scipy.interpolate import interp1d
import scipy.constants as sc
import astropy.constants as ac
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import binned_statistic
from scipy.optimize import minimize

import matplotlib.pyplot as plt

class Surface:

    def __init__(self,
        cube: None,
        PA: float = None,
        inc: float = None,
        dRA: float = 0.0,
        dDec: float = 0.0,
        x_star: float = None,
        y_star: float = None,
        v_syst: float = None,
        sigma: float = 5,
        **kwargs):
        '''
        Parameters
        ----------
        cube
            Instance of an image cube of the line data.
        PA
            Position angle of the source in degrees, measured from east to north.
        inc
            Inclination of the source in degrees.
        dRA
            offset in arcseconds
        dDec
            offset in arcseconds
        x_star
            offset in pixels, set as nx/2 for the center
        y_star
            offset in pixels, set as nx/2 for the center
        v_syst
            system velocity in km/s.
        v_mask
            mask channels within a certain km/s range of the systematic velocity
        sigma
            cutt off threshold to fit surface

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

        '''

        self.cube = cube

        self.PA = PA
        self.inc = inc

        if x_star is None and y_star is None:
            self.x_star = (cube.nx/2 +1) + (dRA*np.pi/(180 * 3600))/np.abs(cube.header['CDELT1']*np.pi/180)
            self.y_star = (cube.ny/2 +1) + (dDec*np.pi/(180 * 3600))/np.abs(cube.header['CDELT2']*np.pi/180)
        else:
            self.x_star = x_star
            self.y_star = y_star

        self.sigma = sigma
        self.v_syst = v_syst

        self._detect_surface()
        self._compute_surface()

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

        cube = self.cube
        nx, nv = cube.nx, cube.nv

        n_surf = np.zeros(nv, dtype=int)
        x_surf = np.zeros([nv,nx])
        y_surf = np.zeros([nv,nx,2])
        Tb_surf = np.zeros([nv,nx,2])
        Ib_surf = np.zeros([nv,nx,2])

        # Measure the rms in 1st channel
        std = np.nanstd(cube.image[1,:,:])

        surface_color = ["red","blue"]

        # Loop over the channels
        for iv in range(nv):
            print(iv,"/",nv-1)
            # Rotate the image so major axis is aligned with x-axis.
            im = np.nan_to_num(cube.image[iv,:,:])
            if self.PA is not None:
                im = np.array(rotate(im, self.PA - 90.0, reshape=False))

            # Setting up arrays in each channel map
            in_surface = np.full(nx,False)
            j_surf = np.zeros([nx,2], dtype=int)
            j_surf_exact = np.zeros([nx,2])
            T_surf = np.zeros([nx,2])
            I_surf = np.zeros([nx,2])

            # Loop over the pixels along the x-axis to find surface
            for i in range(nx):
                vert_profile = im[:,i]
                # find the maxima in each vertical cut, at signal above X sigma
                # ignore maxima not separated by at least a beam
                j_max = search_maxima(vert_profile,threshold=self.sigma*std, dx=cube.bmaj/cube.pixelscale)

                if (j_max.size>1): # We need at least 2 maxima to locate the surface
                    in_surface[i] = True

                    # indices of the back and front side
                    j_surf[i,:] = np.sort(j_max[:2])

                    # exclude maxima that do not make sense
                    if self.y_star is not None:
                        if (j_surf[i,1] < self.y_star):
                            # Houston, we have a pb : the back side of the disk cannot appear below the star
                            j_max_sup = j_max[np.where(j_max > self.y_star)]
                            if j_max_sup.size:
                                j_surf[i,1] = j_max_sup[0]
                                j_surf[i,0] = j_max[0]
                            else:
                                in_surface[i] = False

                        if (np.mean(j_surf[i,:]) < self.y_star):
                            # the average of the top surfaces cannot be below the star
                            in_surface[i] = False

                    #-- We find a spatial quadratic to refine position of maxima (like bettermoment does in velocity)
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
                        T_surf[i,k] = cube._Jybeam_to_Tb(f_max) # Converting to Tb (currently assuming the cube is in Jy/beam)
                        I_surf[i,k] = f_max

            #-- Now we try to clean out a bit the surfaces we have extracted

            #-- We test if front side is too high or the back side too low
            # this happens when the data gets noisy or diffuse and there are local maxima
            # fit a line to average curve and remove points from front if above average
            # and from back surface if  below average (most of these case should have been dealt with with test on star position)

            # could search for other maxima but then it means that data is noisy anyway
            #e.g. measure_surface(HD163, 45, plot=True, PA=-45,plot_cut=503,sigma=10, y_star=478)
            if np.any(in_surface):
                x = np.arange(nx)
                x1 = x[in_surface]

                y1 = np.mean(j_surf_exact[in_surface,:],axis=1)

                if (len(x1) > 2):
                    P = np.polyfit(x1,y1,1)

                    #x_plot = np.array([0,nx])
                    #plt.plot(x_plot, P[1] + P[0]*x_plot)

                    #in_surface_tmp = in_surface &  (j_surf_exact[:,0] < (P[1] + P[0]*x)) # test only front surface
                    in_surface_tmp = in_surface &  (j_surf_exact[:,0] < (P[1] + P[0]*x)) & (j_surf_exact[:,1] > (P[1] + P[0]*x))

                    # We remove the weird point and reddo the fit again to ensure the slope we use is not too bad
                    x1 = x[in_surface_tmp]
                    y1 = np.mean(j_surf_exact[in_surface_tmp,:],axis=1)
                    P = np.polyfit(x1,y1,1)

                    #in_surface = in_surface &  (j_surf_exact[:,0] < (P[1] + P[0]*x)) # test only front surface
                    in_surface = in_surface & (j_surf_exact[:,0] < (P[1] + P[0]*x)) & (j_surf_exact[:,1] > (P[1] + P[0]*x))

                # Saving the data
                n = np.sum(in_surface)
                n_surf[iv] = n # number of points in that surface
                if n:
                    x_surf[iv,:n] = x[in_surface]
                    y_surf[iv,:n,:] = j_surf_exact[in_surface,:]
                    Tb_surf[iv,:n,:] = T_surf[in_surface,:]
                    Ib_surf[iv,:n,:] = I_surf[in_surface,:]

                #-- test if we have points on both side of the star
                # - remove side with the less points

        #--  Additional spectral filtering to clean the data
        self.n_surf = n_surf
        self.x_sky = x_surf
        self.y_sky = y_surf
        self.Tb = Tb_surf
        self.I = Ib_surf
        self.snr = Ib_surf/std


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
        y_f = self.y_sky[:,:,1] - self.y_star   # far side, y[channel number, x index]
        y_n = self.y_sky[:,:,0] - self.y_star   # near side
        y_c = 0.5 * (y_f + y_n)
        #y_c = np.ma.masked_array(y_c,mask).compressed()

        x = self.x_sky - self.x_star
        y = (y_f - y_c) / np.cos(inc_rad)

        r = np.hypot(x,y) # Note : does not depend on y_star
        h = y_c / np.sin(inc_rad)
        v = (self.cube.velocity[:,np.newaxis] - self.v_syst) * r / ((self.x_sky - self.x_star) * np.sin(inc_rad)) # does not depend on y_star

        r *= self.cube.pixelscale
        h *= self.cube.pixelscale

        # -- we remove the points with h<0 (they correspond to values set to 0 in y)
        # and where v is not defined
        mask = (h<0) | np.isinf(v) | np.isnan(v)

        # -- we remove channels that are too close to the systemic velocity
        mask = mask | (np.abs(self.cube.velocity - self.v_syst) < 1.5)[:,np.newaxis]

        r = np.ma.masked_array(r,mask)
        h = np.ma.masked_array(h,mask)
        v = np.ma.masked_array(v,mask)

        # -- If the disc rotates in the opposite direction as expected
        if (np.mean(v) < 0):
            v = -v

        # -- Todo : optimize position, inclination (is that posible without a model ?), PA (need to re-run detect surface)
        self.x = x
        self.y = y
        self.r = r
        self.h = h
        self.v = v


    def plot_surfaces(self,
        nbins: int = 30,
        m_star: float = None,
        ):

        r = self.r
        h = self.h
        v=self.v
        T = np.mean(self.Tb[:,:,:],axis=2)

        r_data = r.ravel().compressed()#[np.invert(mask.ravel())]
        h_data = h.ravel().compressed()#[np.invert(mask.ravel())]
        v_data = v.ravel().compressed()#[np.invert(mask.ravel())]
        T_data = np.mean(self.Tb[:,:,:],axis=2).ravel()[np.invert(r.mask.ravel())]

        #-- fitting a power-law
        P, res_h, _, _, _ = np.ma.polyfit(np.log10(r.ravel()),np.log10(h.ravel()),1, full=True)
        x = np.linspace(np.min(r),np.max(r),100)

        font=30
        line_width=3

        fig = plt.figure(figsize=(30,11))
        gs = fig.add_gridspec(nrows=1,ncols=3)
        gs.update(wspace=0.2, hspace=0.05)
        ax=[]
        for i in range(0,1):
            for j in range(0,3):
                ax.append(fig.add_subplot(gs[i,j]))

        #Altitude

        ax[0].scatter(r.ravel(),h.ravel(),alpha=0.5,s=10,color="grey",marker='o')

        bins, _, _ = binned_statistic(r_data,[r_data,h_data], bins=nbins)
        std, _, _  = binned_statistic(r_data,h_data, 'std', bins=nbins)

        ax[0].errorbar(bins[0,:], bins[1,:],yerr=std, ecolor="red", fmt='o', mec='k', mfc='red', ms=10, elinewidth=2)

        ax[0].plot(x, 10**P[1] * x**P[0], color='k', ls='--', alpha=0.75, lw=line_width)


        #Velocity
        ax[1].scatter(r.ravel(),v.ravel(),alpha=0.5,s=10,color="grey",marker='o', label = 'data')

        bins, _, _ = binned_statistic(r_data,[r_data,v_data], bins=nbins)
        std, _, _  = binned_statistic(r_data,v_data, 'std', bins=nbins)

        ax[1].errorbar(bins[0,:], bins[1,:],yerr=std, ecolor="red", fmt='o', mec='k', mfc='red', ms=10, elinewidth=2)

        if m_star:
            v_model = self._keplerian_disc(m_star)
            ax[1].scatter(r_data, v_model,alpha=0.5,s=10,color="blue",marker='o', label = 'model')

        #Temperature

        ax[2].scatter(r.ravel(),T.ravel(),alpha=0.5,s=10,color="grey",marker='o')

        bins, _, _ = binned_statistic(r_data,[r_data,T_data], bins=nbins)
        std, _, _  = binned_statistic(r_data,T_data, 'std', bins=nbins)

        ax[2].errorbar(bins[0,:], bins[1,:],yerr=std, ecolor="red", fmt='o', mec='k', mfc='red', ms=10, elinewidth=2)

        ax[0].set_ylabel('Height (")', fontsize=font)
        ax[1].set_ylabel('Velocity (km/s)', fontsize=font)
        ax[2].set_ylabel('Brightness Temperature (K)', fontsize=font)

        ax[0].set_xlabel('Radius (")', fontsize=font)
        ax[1].set_xlabel('Radius (")', fontsize=font)
        ax[2].set_xlabel('Radius (")', fontsize=font)

        ax[0].tick_params(axis='both', direction='out', labelbottom=True, labelleft=True, top=False, right=False, width=3, length=8 ,labelsize=font-10)
        ax[1].tick_params(axis='both', direction='out', labelbottom=True, labelleft=True, top=False, right=False, width=3, length=8 ,labelsize=font-10)
        ax[2].tick_params(axis='both', direction='out', labelbottom=True, labelleft=True, top=False, right=False, width=3, length=8 ,labelsize=font-10)


        #Adding hard outline
        bar_size = 3
        c ="black"
        for i, axes in enumerate(ax):
            ax[i].axhline(linewidth=bar_size, y=ax[i].get_ylim()[0], color=c)
            ax[i].axvline(linewidth=bar_size, x=ax[i].get_xlim()[0], color=c)
            ax[i].axhline(linewidth=bar_size, y=ax[i].get_ylim()[1], color=c)
            ax[i].axvline(linewidth=bar_size, x=ax[i].get_xlim()[1], color=c)


        return P

    def plot_channel(self,iv, win=20,ax=None):

        if ax is None:
            ax = plt.gca()

        cube = self.cube
        x = self.x_sky
        y = self.y_sky
        n_surf = self.n_surf

        im = np.nan_to_num(cube.image[iv,:,:])
        if self.PA is not None:
            im = np.array(rotate(im, self.PA - 90.0, reshape=False))

        ax.imshow(im, origin="lower")#, interpolation="bilinear")

        if n_surf[iv]:
            ax.plot(x[iv,:n_surf[iv]],y[iv,:n_surf[iv],0],"o",color="red",markersize=1)
            ax.plot(x[iv,:n_surf[iv]],y[iv,:n_surf[iv],1],"o",color="blue",markersize=1)
            #plt.plot(x,np.mean(y,axis=1),"o",color="white",markersize=1)

            # We zoom on the detected surfaces
            #ax.set_xlim(np.min(x[iv,:n_surf[iv]]) - 10*cube.bmaj/cube.pixelscale,np.max(x[iv,:n_surf[iv]]) + 10*cube.bmaj/cube.pixelscale)
            #ax.set_ylim(np.min(y[iv,:n_surf[iv],:]) - 10*cube.bmaj/cube.pixelscale,np.max(y[iv,:n_surf[iv],:]) + 10*cube.bmaj/cube.pixelscale)


    def plot_channels(self,n=20, win=21):

        nv = self.cube.nv
        dv = np.floor(nv/n).astype(int)

        ncols=5
        nrows = np.ceil(n / ncols).astype(int)

        fig, axs = plt.subplots(ncols=5, nrows=nrows, figsize=(11, 2*nrows+1),constrained_layout=True,num=win)

        for i, ax in enumerate(axs.flatten()):
            self.plot_channel(i*dv,ax=ax)

    def fit_central_mass(self,
        initial_guess: float = None,
        dist: float = None):
        '''
        Parameters
        ----------
        initial_guess
            initial guess of central mass
        dist
            distance of source in pc.

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

        '''

        if (dist is None):
            raise ValueError("dist must be provided")

        initial = np.array([initial_guess])

        soln = minimize(self._ln_like, initial, args=(dist), bounds=((0, None),))

        print("Maximum likelihood estimate:")
        print(soln)

        return soln.x

    def _ln_like(self, theta, dist):
        """Compute the ln like..
        """

        #define param values
        m_star = theta[0]

        # compute the model for the chi2
        v_model = self._keplerian_disc(m_star, dist)

        v = self.v.ravel().compressed()

        #using 1/snr as the error on the velocity
        v_error = 1/(np.mean(self.snr[:,:,:],axis=2).ravel()[np.invert(self.r.mask.ravel())])

        # chi2
        chi2= np.sum(((v - v_model)**2 / v_error**2) +  np.log(2*np.pi*v_error**2))
        return 0.5 * chi2

    def _keplerian_disc(self, m_star, dist):
        """M_star for a kerplerian disc"""
        #Defining constants
        G = sc.G
        msun = ac.M_sun.value


        r = self.r.ravel().compressed() * dist * sc.au
        h = self.h.ravel().compressed() * dist * sc.au
        v = np.sqrt((G*m_star*msun*r**2)/((r**2 + h**2)**(3/2)))/1000
        return v


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
