#HD163 = casa.Cube("/Users/cpinte/Observations/HD163/ALMA/Isella/mine/HD163296_CO_100m.s-1.image.fits.gz")
#measure_surface(HD163, 61, plot=True, PA=-45,plot_cut=534,sigma=10, y_star=478)

from scipy.ndimage import rotate, shift
from scipy.interpolate import interp1d
import scipy.constants as sc
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import binned_statistic



def measure_mol_surface(cube, n, x, y, T, inc=None, x_star=None, y_star=None, v_syst= None, plot=None, distance=None, optimize_x_star=None):
    """
    inc (float): Inclination of the source in [degrees].
    """

    inc_rad = np.radians(inc)

    #-- Computing the radius and height for each point
    y_c = 0.5 * (np.sum(y[:,:,:], axis=2)) - y_star
    y_f = y[:,:,1] - y_star
    y_n = y[:,:,0] - y_star

    r = np.hypot(x - x_star, (y_f - y_c) / np.cos(inc_rad)) # does not depend on y_star
    h = y_c / np.sin(inc_rad)
    v = (cube.velocity[:,np.newaxis] - v_syst) * r / ((x - x_star) * np.sin(inc_rad)) # does not depend on y_star

    if distance is not None:
        r *= cube.pixelscale * distance
        h *= cube.pixelscale * distance

    # value of x_star plays on dispersion of h and v
    # value of PA creates a huge dispersion
    # value of y_star shift the curve h(r) vertically, massive change on flaring exponent
    # value of inc changes the shape a bit, but increases dispersion on velocity

    # -- we remove the points with h<0 (they correspond to values set to 0 in y)
    # and where v is not defined
    mask = (h<0) | np.isinf(v) | np.isnan(v)

    # -- we remove channels that are too close to the systemic velocity
    mask = mask | (np.abs(cube.velocity - v_syst) < 1.5)[:,np.newaxis]

    print(r.shape, mask.shape)

    r = np.ma.masked_array(r,mask)
    h = np.ma.masked_array(h,mask)
    v = np.ma.masked_array(v,mask)


    # -- If the disc rotates in the opposite direction as expected
    if (np.mean(v) < 0):
        v = -v

    plt.figure(30)
    plt.clf()
    plt.scatter(r.ravel(),h.ravel(),alpha=0.2,s=5)
    plt.show()

    plt.figure(31)
    plt.clf()
    plt.scatter(r.ravel(),v.ravel(),alpha=0.2,s=5)
    plt.show()

    #-- Ignoring channels close to systemic velocity

    #-- fitting a power-law
    P, res_h, _, _, _ = np.ma.polyfit(np.log10(r.ravel()),np.log10(h.ravel()),1, full=True)
    x = np.linspace(np.min(r),np.max(r),100)
    plt.figure(30)
    plt.plot(x, 10**P[1] * x**P[0])
    print(P, "res", res_h)


    r_data = r.ravel()[np.invert(mask.ravel())]
    h_data = h.ravel()[np.invert(mask.ravel())]
    v_data = v.ravel()[np.invert(mask.ravel())]

    #plt.scatter(r_data,h_data)

    bins, _, _ = binned_statistic(r_data,[r_data,h_data], bins=30)
    std, _, _  = binned_statistic(r_data,h_data, 'std', bins=30)

    print("STD =", np.median(std))
    plt.errorbar(bins[0,:], bins[1,:],yerr=std, color="red")

    bins_v, _, _ = binned_statistic(r_data,[r_data,v_data], bins=30)
    std_v, _, _  = binned_statistic(r_data,v_data, 'std', bins=30)

    print("STD =", np.median(std_v))  # min seems a better estimate for x_star than std_h
    plt.figure(31)
    plt.errorbar(bins_v[0,:], bins_v[1,:], yerr=std_v, color="red", marker="o", fmt=' ', markersize=2)

    # -- Optimize position, inclination (is that posible without a model ?), PA (need to re-run detect surface)
    return r, h, v



def detect_surface(cube, PA=None, plot=False, plot_cut=None, sigma=None, y_star=None, win=20):
    """
    Infer the upper emission surface from the provided cube
    extract the emission surface in each channel and loop over channels

    Args:
        cube (casa instance): An imgcube instance of the line data.

        PA (float): Position angle of the source in [degrees].
        y_star (optional) : position of star in  pixel (in rorated image), used to filter some bad data
        without y_star, more points might be rejected
    """

    nv = cube.nv
    nx = cube.nx

    n_surf = np.zeros(nv, dtype=int)
    x_surf = np.zeros([nv,nx])
    y_surf = np.zeros([nv,nx,2])
    Tb_surf = np.zeros([nv,nx,2])

    # Measure the rms in 1st channel
    std = np.nanstd(cube.image[1,:,:])

    surface_color = ["red","blue"]


    for iv in range(nv):
        print(iv,"/",nv-1)
        # Rotate the image so major axis is aligned with x-axis.
        im = np.nan_to_num(cube.image[iv,:,:])
        if PA is not None:
            im = np.array(rotate(im, PA - 90.0, reshape=False))

        # plot channel map as bakcground
        if plot:
            pdf = PdfPages('CO_layers.pdf')
            plt.figure(win)
            plt.clf()
            plt.imshow(im, origin="lower")#, interpolation="bilinear")

        #-- Loop over x pixel axis to find surface
        in_surface = np.full(nx,False)
        j_surf = np.zeros([nx,2], dtype=int)
        j_surf_exact = np.zeros([nx,2])
        T_surf = np.zeros([nx,2])


        for i in range(nx):
            # find the maxima in each vertical cut, at signal above X sigma
            # ignore maxima not separated by at least a beam
            vert_profile = im[:,i]
            j_max = search_maxima(vert_profile,threshold=sigma*std, dx=cube.bmaj/cube.pixelscale)

            if (j_max.size>1): # We need at least 2 maxima to locate the surface
                in_surface[i] = True

                # indices of the back and front side
                j_surf[i,:] = np.sort(j_max[:2])

                if y_star is not None:
                    if (j_surf[i,1] < y_star):
                        # Houston, we have a pb : the back side of the disk cannot appear below the star
                        j_max_sup = j_max[np.where(j_max > y_star)]
                        if j_max_sup.size:
                            j_surf[i,1] = j_max_sup[0]
                            j_surf[i,0] = j_max[0]
                        else:
                            in_surface[i] = False

                    if (np.mean(j_surf[i,:]) < y_star):
                        # the average of the top surfaces cannot be below the star
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
            P = np.polyfit(x1,y1,1)

            #x_plot = np.array([0,nx])
            #plt.plot(x_plot, P[1] + P[0]*x_plot)

            in_surface_tmp = in_surface &  (j_surf_exact[:,0] < (P[1] + P[0]*x)) # test only front surface
            in_surface_tmp = in_surface &  (j_surf_exact[:,0] < (P[1] + P[0]*x)) & (j_surf_exact[:,1] > (P[1] + P[0]*x))

            # We remove the weird point and reddo the fit again to ensure the slope we use is not too bad
            x1 = x[in_surface_tmp]
            y1 = np.mean(j_surf_exact[in_surface_tmp,:],axis=1)
            P = np.polyfit(x1,y1,1)

            #in_surface = in_surface &  (j_surf_exact[:,0] < (P[1] + P[0]*x)) # test only front surface
            in_surface = in_surface &  (j_surf_exact[:,0] < (P[1] + P[0]*x)) & (j_surf_exact[:,1] > (P[1] + P[0]*x))

            # Saving the data
            n = np.sum(in_surface)
            n_surf[iv] = n # number of points in that surface
            if n:
                x_surf[iv,:n] = x[in_surface]
                y_surf[iv,:n,:] = j_surf_exact[in_surface,:]
                Tb_surf[iv,:n,:] = T_surf[in_surface,:]

                # We plot  the detected points
                if plot:
                    plt.figure(win)
                    plt.plot(x_surf,y_surf[:,0],"o",color=surface_color[0],markersize=1)
                    plt.plot(x_surf,y_surf[:,1],"o",color=surface_color[1],markersize=1)
                    #plt.plot(x_surf,np.mean(y_surf,axis=1),"o",color="white",markersize=1)

                    # We zoom on the detected surfaces
                    plt.xlim(np.min(x_surf) - 10*cube.bmaj/cube.pixelscale,np.max(x_surf) + 10*cube.bmaj/cube.pixelscale)
                    plt.ylim(np.min(y_surf) - 10*cube.bmaj/cube.pixelscale,np.max(y_surf) + 10*cube.bmaj/cube.pixelscale)

                    pdf.savefig()
                    plt.close()

            #-- test if we have points on both side of the star
            # - remove side with the less points


    #--  Additional spectral filtering to clean the data

    return n_surf, x_surf, y_surf, Tb_surf

    #measure_surface(HD163, 50, plot=True, PA=-45,plot_cut=534,sigma=10, y_star=478)




def plot_surface(cube, n, x, y, Tb, iv, PA=None, win=20):

    im = np.nan_to_num(cube.image[iv,:,:])
    if PA is not None:
        im = np.array(rotate(im, PA - 90.0, reshape=False))

    plt.figure(win)
    plt.clf()
    plt.imshow(im, origin="lower")#, interpolation="bilinear")

    if n[iv]:
        plt.plot(x[iv,:n[iv]],y[iv,:n[iv],0],"o",color="red",markersize=1)
        plt.plot(x[iv,:n[iv]],y[iv,:n[iv],1],"o",color="blue",markersize=1)
        #plt.plot(x,np.mean(y,axis=1),"o",color="white",markersize=1)

        # We zoom on the detected surfaces
        plt.xlim(np.min(x[iv,:n[iv]]) - 10*cube.bmaj/cube.pixelscale,np.max(x[iv,:n[iv]]) + 10*cube.bmaj/cube.pixelscale)
        plt.ylim(np.min(y[iv,:n[iv],:]) - 10*cube.bmaj/cube.pixelscale,np.max(y[iv,:n[iv],:]) + 10*cube.bmaj/cube.pixelscale)




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
