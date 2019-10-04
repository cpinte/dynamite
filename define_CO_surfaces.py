import astropy.io.fits as fits
import astropy.constants as const
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.ndimage.interpolation import rotate


class CASA_image:
    pass

class CO_surface:
    nv = int()
    i = np.zeros(200)
    j = np.zeros(200)

#------------------------------------------------------

source = "HD97048"


if source == "HD97048":
    Mstar = 2.5
    fits_names = ["/Users/cpinte/Observations/HAeBe/HD97048/Band6/HD_97048_12CO_21_uniform_image.image.fits","/Users/cpinte/Observations/HAeBe/HD97048/Band6/HD_97048_13CO_21_uniform_image.image.fits","/Users/cpinte/Observations/HAeBe/HD97048/Band6/HD_97048_C18O_21_uniform_image.image.fits"]  # 200m/s]
    which_CO = [12,13,18]
    color = ["blue","red","green"]
    x_star = 377 ; y_star = 376
    PA = -96
    inc = 41.
    distance = 158.
    Vsyst = 4750.  #m/s
    vCO_test = 0 ; delta_vCO_test = 100000 ; ivtest = 85 ; clear=1
    xmin = 50 ; xmax = 400.


fits_name = fits_names[0]

hdu = fits.open(fits_name)


nx = hdu[0].header['NAXIS1']
ny = hdu[0].header['NAXIS2']
nv = hdu[0].header['NAXIS3']
CRPIX1 = hdu[0].header['CRPIX1']
CRPIX2 = hdu[0].header['CRPIX2']
CDELT1 = hdu[0].header['CDELT1']
pix_size = abs(CDELT1) * 3600  # arcsec

BMIN = hdu[0].header['BMIN']
BMAJ = hdu[0].header['BMAJ']
BPA = hdu[0].header['BPA']

CDELT3 = hdu[0].header['CDELT3']
CRVAL3 = hdu[0].header['CRVAL3']
CRPIX3 = hdu[0].header['CRPIX3']
restfreq = hdu[0].header['RESTFRQ']

CO = hdu[0].data


dv = const.c.value * abs(CDELT3)/CRVAL3 ;
nu = CRVAL3 + CDELT3 * (np.arange(1,nv+1) - CRPIX3) ;
wl = const.c.value/CRVAL3 ;

vCO = -(nu - restfreq)/restfreq



freqs = np.arange(2, 20, 3)





def define_surface(data):

    class Window_Interface():

        def __init__(self,data):
            self.fig  =plt.gcf()
            self.ax = plt.gca()
            self.im = data
            self.ind = 0
            self.angle = 0.
            self._plot_channel(first_time=True)

        def next(self, event):
            self.ind += 1
            self._plot_channel()

        def prev(self, event):
            self.ind -= 1
            self._plot_channel()

        def next10(self, event):
            self.ind += 10
            self._plot_channel()

        def prev10(self, event):
            self.ind -= 10
            self._plot_channel()

        def rotate_plus10(self, event):
            self.angle += 10
            self._plot_channel()

        def rotate_moins10(self, event):
            self.angle -= 10
            self._plot_channel()

        def rotate_plus(self, event):
            self.angle += 1
            self._plot_channel()

        def rotate_moins(self, event):
            self.angle -= 1
            self._plot_channel()

        def create_polygon(self, event):
            polygon = Polygon_Builder(fig,ax)
            self._connect_mouse()

        def _plot_channel(self,first_time=False):
            print(self.ind)
            plt.sca(self.ax)
            xmin, xmax = self.ax.get_xlim()
            ymin, ymax = self.ax.get_ylim()
            plt.cla()
            plt.imshow(rotate(self.im[0,self.ind,:,:], self.angle, reshape=False),origin='lower')
            if not first_time: # We keep the same limits
                self.ax.set_xlim(xmin, xmax)
                self.ax.set_ylim(ymin, ymax)

            plt.text(0.0,1.05, f"Current channel: {self.ind}",transform=self.ax.transAxes)
            plt.text(0.5,1.05, f"Current rot: {self.angle}",transform=self.ax.transAxes)
            plt.draw()
            print("CHANNEL PLOT DONE")

        def _connect_mouse(self):
            print("Connecting button")
            self.cid = self.fig.canvas.mpl_connect('button_press_event', self._button_press)

        def _button_press(self,event):
            print("")
            print('click', event)
            print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                  ('double' if event.dblclick else 'single', event.button,
                   event.x, event.y, event.xdata, event.ydata))
            if event.inaxes != self.ax:
                print("OUT !!!!!!!!!") # Ok, we do not anything if we are in the right axis
                return

            if event.button == 1: # Adding points
                self.npoints += 1
                x = event.xdata
                y = event.ydata
                self.x.append(x)
                self.y.append(y)
                plt.sca(self.ax)
                plt.plot(x,y,"o",color="magenta",alpha=0.5)
                if self.npoints > 1:
                    plt.plot(self.x[-2:],self.y[-2:],color="magenta",alpha=0.5)
                plt.draw()
            else:
                if self.npoints > 0:
                    self.npoints -= 1
                    # Looking for the closest point
                    x = event.xdata
                    y = event.ydata

                    d2 = (np.array(self.x)-x)**2 + (np.array(self.y)-y)**2


    class Polygon:
        def __init__(self):
            self.x = []
            self.y = []
            self.npoints = 0

        def add_point(self):


        def delete_point(self):
           pass

    print("BIBI is confused")



    print("BIBI is still confused")


    # -- We define on the window interface on the data
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    callback = Window_Interface(data)
    print("WE ARE HERE")

    # -- We add the interface buttons

    #-- Velocity buttons
    axnext = plt.axes([0.81, 0.03, 0.1, 0.05])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)
    axnext._bnext = bnext # Creating dummy references to make buttons available

    axprev = plt.axes([0.7, 0.03, 0.1, 0.05])
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)
    axprev._bprev = bprev

    axnext10 = plt.axes([0.81, 0.09, 0.1, 0.05])
    bnext10 = Button(axnext10, '+ 10')
    bnext10.on_clicked(callback.next10)
    axnext10._bnext10 = bnext10

    axprev10 = plt.axes([0.7, 0.09, 0.1, 0.05])
    bprev10 = Button(axprev10, '- 10')
    bprev10.on_clicked(callback.prev10)
    axprev10._bprev10 = bprev10

    #-- Rotation buttons
    axrotp = plt.axes([0.1, 0.03, 0.1, 0.05])
    brotp = Button(axrotp, '+1$^o$')
    brotp.on_clicked(callback.rotate_plus)
    axrotp._brotp = brotp

    axrotm = plt.axes([0.21, 0.03, 0.1, 0.05])
    brotm = Button(axrotm, '-1$^o$')
    brotm.on_clicked(callback.rotate_moins)
    axrotm._brotm = brotm

    axrotp10 = plt.axes([0.1, 0.09, 0.1, 0.05])
    brotp10 = Button(axrotp10, '+10$^o$')
    brotp10.on_clicked(callback.rotate_plus10)
    axrotp10._brotp10 = brotp10

    axrotm10 = plt.axes([0.21, 0.09, 0.1, 0.05])
    brotm10 = Button(axrotm10, '-10$^o$')
    brotm10.on_clicked(callback.rotate_moins10)
    axrotm10._brotm10 = brotm10

    #-- Polygon button
    axpoly = plt.axes([0.4, 0.075, 0.2, 0.05])
    b_poly = Button(axpoly, "define surface")
    b_poly.on_clicked(callback.create_polygon)
    axpoly._bpoly = b_poly

    plt.sca(ax)
