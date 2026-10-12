Basic Usage
==========

This guide demonstrates the basic usage of the dynamite package to analyze molecular line data from a protoplanetary disk.

Loading and Processing Data
-------------------------

First, import the necessary modules and create a Surface instance:

.. code-block:: python

    from dynamite.measure_height import Surface
    import matplotlib.pyplot as plt

    # Create a Surface instance with your data cube
    surface = Surface(
        cube="path/to/your/cube.fits",  # Path to your FITS cube
        PA=45.0,                        # Position angle in degrees
        inc=60.0,                       # Inclination in degrees
        dist=140.0,                     # Distance to source in parsecs
        sigma=10.0,                     # Detection threshold
        plot=True                       # Enable plotting
    )

The Surface class will automatically:
1. Load and process your data cube
2. Extract emission surfaces
3. Compute disk parameters
4. Generate diagnostic plots

Analyzing Results
---------------

You can access various properties of the analyzed data:

.. code-block:: python

    # Get radius, height, and velocity arrays
    r = surface.r
    h = surface.h
    v = surface.v

    # Plot the results
    surface.plot_surfaces(
        nbins=30,                    # Number of bins for averaging
        m_star=1.0,                  # Central stellar mass in Msun
        dist=140.0,                  # Distance in parsecs
        plot_power_law=True,         # Fit and plot power law
        r0=1.0                       # Reference radius for power law
    )

Fitting Surface Height
--------------------

You can fit the surface height with different models:

.. code-block:: python

    # Fit with a power law
    params, cov = surface.fit_surface_height(
        r0=1.0,                      # Reference radius
        tapered_power_law=False      # Use simple power law
    )

    # Fit with a tapered power law
    params, cov = surface.fit_surface_height(
        r0=1.0,
        tapered_power_law=True
    )

    # Fit with a Gaussian Process
    t, mu, std = surface.fit_surface_height_gp()

Estimating Central Mass
---------------------

To estimate the central stellar mass:

.. code-block:: python

    surface.fit_central_mass(
        initial_guess=1.0,           # Initial mass guess in Msun
        dist=140.0                   # Distance in parsecs
    )
    print(f"Estimated central mass: {surface.m_star:.2f} Msun")

Converting to MCFOST Format
-------------------------

If you need to convert the results to MCFOST format:

.. code-block:: python

    mcfost_inc, r, az = surface.to_mcfost(
        planet_r=0.0,                # Planet radius in arcsec
        planet_PA=0.0                # Planet position angle in degrees
    )
    print(f"MCFOST inclination: {mcfost_inc:.2f} deg")
    print(f"MCFOST planet radius: {r:.2f} au")
    print(f"MCFOST planet azimuth: {az:.2f} deg") 