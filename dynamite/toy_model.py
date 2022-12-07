import numpy as np
import scipy.constants as sc
from astropy import units as u


class toy_model:

    def __init__(self, Mstar=None, dist=None, inc=None, PA=None, FOV=None, npix=None, cube=None, vlsr=0.,
                 z_func=None, r0=None, z0=None, beta=None, lower_surface=False):

        # Testing all the arguments
        if z_func is None:
            if r0 is None or z0 is None or beta is None:
                raise ValueError("r0, z0 and beta must be provided in zfunc is None")
            def z_func(r):
                return z0 * (r/r0)**beta

        if Mstar is None:
            raise ValueError("'Mstar' must be provided.")
        self.Mstar = Mstar

        if dist is None:
            raise ValueError("'Mstar' must be provided.")
        self.dist = dist

        if inc is None:
            raise ValueError("'inc' must be provided.")
        self.inc = inc
        self.sin_i = np.sin(np.radians(inc))

        if PA is None:
            raise ValueError("'PA' must be provided.")
        self.PA = PA

        if cube is not None:
            FOV, npix = cube.FOV, cube.nx
            self.velocity = cube.velocity

        if FOV is None:
            raise ValueError("'FOV' or 'cube' must be provided.")
        self.FOV = FOV

        if npix is None:
            raise ValueError("'npix' or 'cube' must be provided.")
        self.npix = npix

        # Axis in arcsec
        self.yaxis = np.linspace(-self.FOV/2,self.FOV/2,num=self.npix)
        self.xaxis = -self.yaxis

        self.extent = [self.FOV/2,-self.FOV/2,-self.FOV/2,self.FOV/2]

        # Sky coordinates
        self.x_sky, self.y_sky = np.meshgrid(self.xaxis, self.yaxis)

        # Model coordinates in arcsec
        self.x_disk, self.y_disk, self.z_disk = self.sky_to_surface(inc=inc, PA=PA, z_func=z_func, lower_surface=lower_surface)

        # Model coordinates in au
        self.x_disk *= dist
        self.y_disk *= dist
        self.z_disk *= dist

        self.r_disk = np.hypot(self.x_disk,self.y_disk)
        self.theta_disk = np.arctan2(self.y_disk, self.x_disk)

        # Model velocity
        self.v_Kep = self.Keplerian_velocity(Mstar=self.Mstar, r=self.r_disk, z=self.z_disk) / 1000. #km/s
        self.v_proj = self.v_Kep * np.cos(self.theta_disk) * self.sin_i + vlsr


    def Keplerian_velocity(self, Mstar=None, r=None, z=0):
        """
        Calculate the Keplerian velocity field, including vertical shear in [m/s]

        Args:
            r (array): Midplane radii in [au].
            z (Optional[array]): Height of emission surface in [au].
        """

        Msun = u.Msun.to(u.kg)

        if Mstar is None:
            raise ValueError("'Mstar' must be provided.")
        if r is None:
            raise ValueError("'r' must be provided.")

        return  np.sqrt(sc.G * Mstar * Msun * (r**2 / np.hypot(r,z)**3) / sc.au)

    def sky_to_midplane(self, x_sky=None, y_sky=None, inc=None, PA=None):
        """Return the coordinates (x,y) of the midplane in arcsec"""

        #--  De-rotate (x_sky, y_sky) by PA [deg] to make the disk major axis horizontal
        PA_rad = np.radians(PA)
        cos_PA, sin_PA = np.cos(PA_rad), np.sin(PA_rad)
        x_rot =  x_sky * sin_PA + y_sky * cos_PA
        y_rot =  x_sky * cos_PA - y_sky * sin_PA

        #-- Deproject for inclination
        return x_rot, y_rot / np.cos(np.radians(inc))

    def sky_to_surface(self, inc=None, PA=None, z_func=None, lower_surface=False):
        """Return coordinates (x,y,z) of surface in arcsec"""

        n_iter = 20
        tan_i = np.tan(np.radians(inc))
        if lower_surface:
            z_factor = -1
        else:
            z_factor = 1

        x_mid, y_mid = self.sky_to_midplane(x_sky=self.x_sky, y_sky=self.y_sky, inc=inc, PA=PA)
        x_mid2 = x_mid**2

        #-- We solve the z and y iteratively
        # Alternatively, we could re-use the second degree equation below if there is a simple prescription for z and no warp
        # Todo : while loop and test on precision
        x, y = x_mid, y_mid
        for i in range(n_iter):
            r = np.sqrt(x_mid2 + y**2)
            z = z_factor * z_func(r) #+ w_func(r, np.arctan2(y, x))
            y = y_mid + z * tan_i

        return x, y, z

    def plot_isovelocity_curve(self, v=None, channel=None, ax=None,
                               rmin=None, rmax=None, nearside_only=False, farside_only=False,
                               correct_velocity=1.0, linestyles="-", flip_v=False,
                               **kwargs):

        if ax is None:
            ax = plt.gca()

        if v is None:
            v = self.velocity[channel]

        #-- Selecting parts of the isovelocity curve
        mask = 1
        if rmin is not None:
            mask = np.where(self.r_disk > rmin, 1.0, np.nan)
        if rmax is not None:
            mask = np.where(self.r_disk < rmax, mask, np.nan)
        if nearside_only:
            mask = np.where(self.y_disk > 0, mask, np.nan)
        if farside_only:
            mask = np.where(self.y_disk < 0, mask, np.nan)

        if flip_v:
            correct_velocity *= -1

        return ax.contour(self.xaxis, self.yaxis, self.v_proj * mask, [v * correct_velocity], linestyles=linestyles, **kwargs)

#--- Old yorick routine translated to python (yorick routine was used for HD163296 paper)
def yorick_toy_model(Mstar, inc, psi, nx=1000, ny=1000, xmax=1000, ymax=1000):

    X = np.linspace(-xmax,xmax,nx)
    Y = np.linspace(-ymax,ymax,ny)

    for i in range(nx):
        for j in range(ny):

            xp = X[i]
            yp = Y[j]

        if ((np.abs(xp) < 1e-6) and (np.abs(xp) < 1e-6)):
            # point on star
            vobs_sup[i,j] = 0.
            vobs_inf[i,j] = 0.
        else:
            a = np.cos(2*inc) + np.cos(2*psi)
            b = -4 * yp * np.tan(inc) * np.sin(psi)**2
            c = -2 * np.sin(psi)**2 * (xp**2 + (yp/np.cos(inc))**2)

            delta = b**2-4*a*c

            t1 = (-b + np.sqrt(delta)) / (2*a)
            t2 = (-b - np.sqrt(delta)) / (2*a)

            # Point sur la surface superieure
            x = xp
            y = yp/np.cos(inc) + t1 * np.sin(inc)
            z = t1 * np.cos(inc)

            theta = np.arctan2(y,x)
            vobs_sup[i,j] = np.sqrt(sc.G * Mstar * u.Msun.to(u.kg) / (np.sqrt(x**2 + y**2)*sc.au)) * np.sin(inc) * np.cos(theta)

            # Point sur la surface inferieure
            x = xp
            y = yp/np.cos(inc) + t2 * np.sin(inc)
            z = t1 * np.cos(inc)

            theta = np.arctan2(y,x)
            vobs_inf[i,j] = np.sqrt(sc.G * Mstar * u.Msun.to(u.kg) / (np.sqrt(x**2 + y**2)*sc.au)) * np.sin(inc) * np.cos(theta)
