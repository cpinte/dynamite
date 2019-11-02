import numpy as np
from astropy import units as u


def toy_model(Mstar, inc, psi, filename, nx=1000, ny=1000, xmax=1000, ymax=1000):

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
