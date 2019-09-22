#HD163 = casa.Cube("/Users/cpinte/Observations/HD163/ALMA/Isella/mine/HD163296_CO_100m.s-1.image.fits.gz")
#measure_surface(HD163, 61, plot=True, PA=-45,plot_cut=534,sigma=10, y_star=478)

from scipy.ndimage import rotate, shift
from scipy.interpolate import interp1d
import numpy as np


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




def measure_surface(cube, channel, PA=None, plot=False, plot_cut=None, sigma=5, y_star=None):
    """
    Infer the emission surface from the provided cube.

    Args:
        cube (casa instance): An imgcube instance of the line data.
        inc (float): Inclination of the source in [degrees].
        PA (float): Position angle of the source in [degrees].
    """

    # Rotate the image so major axis is aligned with x-axis.
    im = np.nan_to_num(cube.image[channel,:,:])
    if PA is not None:
        im = np.array(rotate(im, PA - 90.0, reshape=False))

    # plot channel map as bakcground
    if plot:
        plt.figure(20)
        plt.clf()
        plt.imshow(im, origin="lower")#, interpolation="bilinear")
        plt.show()

    # Measure the rms in 1st channel
    std = np.nanstd(cube.image[1,:,:])

    surface_color = ["red","blue"]

    #-- Loop over x pixel axis to find surface
    nx = cube.nx
    nv = cube.nv
    in_surface = np.full(nx,False)
    j_surf_exact = np.zeros([nv,nx,2])

    for i in range(nx):
        # find the maxima in each vertical cut, at signal above 5 sigma
        # ignore maxima not separated by at least a beam
        vert_profile = im[:,i]
        j_max = search_maxima(vert_profile,threshold=sigma*std, dx=cube.bmaj/cube.pixelscale)

        if (j_max.size>1): # We need at least 2 maxima to locate the surface
            in_surface[i] = True

            # indices of the back and front side
            j_surf = np.sort(j_max[:2])

            if y_star is not None:
                if (j_surf[1] < y_star):
                    # Houston, we have a pb : the back side of the disk cannot appear below the star
                    j_max_sup = j_max[np.where(j_max > y_star)]
                    if j_max_sup.size:
                        j_surf[1] = j_max_sup[0]
                        j_surf[0] = j_max[0]
                    else:
                        in_surface[i] = False

        # - We have found points in the 2 surfaces
        if in_surface[i]:
            if plot_cut:
                if plot_cut==i:
                    plt.figure(21)
                    plt.clf()
                    plt.plot(im[:,i])
                    plt.plot(im[:,i],"o", markersize=1)
                    plt.plot(j_max[0],im[j_max[0],i],"o",color=color[0])
                    plt.plot(j_max[1],im[j_max[1],i],"o",color=color[1])

            #-- We find a spatial quadratic to refine position of maxima(like bettermoment does in velocity)
            for k in range(2):
                j = j_surf[k]

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
                j_surf_exact[channel,i,k] = y_max

    #-- Now we try to clean out a bit the surfaces we have extracted

    #-- We test if front side is too high or the back side too low
    # this happens when the data gets noisy or diffuse and there are local maxima
    # fit a line to average curve and remove points from front if above average
    # and from back surface if  below average (most of these case should have been dealt with with test on star position)

    # could search for other maxima but then it means that data is noisy anyway
    #e.g. measure_surface(HD163, 45, plot=True, PA=-45,plot_cut=503,sigma=10, y_star=478)
    x = np.arange(nx)
    x_surf = x[in_surface]
    y_surf = np.mean(j_surf_exact[channel,in_surface,:],axis=1)
    P = np.polyfit(x_surf,y_surf,1)

    #x_plot = np.array([0,nx])
    #plt.plot(x_plot, P[1] + P[0]*x_plot)

    in_surface_tmp = in_surface &  (j_surf_exact[channel,:,0] < (P[1] + P[0]*x)) # test only front surface
    in_surface_tmp = in_surface &  (j_surf_exact[channel,:,0] < (P[1] + P[0]*x)) & (j_surf_exact[channel,:,1] > (P[1] + P[0]*x))

    # We remove the weird point and reddo the fit again to ensure the slope we use is not too bad
    x_surf = x[in_surface_tmp]
    y_surf = np.mean(j_surf_exact[channel,in_surface_tmp,:],axis=1)
    P = np.polyfit(x_surf,y_surf,1)

    #in_surface = in_surface &  (j_surf_exact[channel,:,0] < (P[1] + P[0]*x)) # test only front surface
    in_surface = in_surface &  (j_surf_exact[channel,:,0] < (P[1] + P[0]*x)) & (j_surf_exact[channel,:,1] > (P[1] + P[0]*x))

    # We plot  the detected points
    if plot:
        plt.figure(20)
        x_surf = x[in_surface]
        y_surf = j_surf_exact[channel,in_surface,:]
        plt.plot(x_surf,y_surf[:,0],"o",color=surface_color[0],markersize=1)
        plt.plot(x_surf,y_surf[:,1],"o",color=surface_color[1],markersize=1)
        #plt.plot(x_surf,np.mean(y_surf,axis=1),"o",color="white",markersize=1)

    return x_surf, y_surf
    #-- test if we have points on both side of the star
    # - remove side with the less points
    #measure_surface(HD163, 50, plot=True, PA=-45,plot_cut=534,sigma=10, y_star=478)
