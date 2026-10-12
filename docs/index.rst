Welcome to dynamite's documentation!
================================

Dynamite (DYNamical Analysis and MultIscale Tomography of line Emission) is a Python package for performing kinematic and tomographic analysis of sub-mm molecular data of protoplanetary disks. It follows the method presented in Pinte et al. 2018, with various improvements, to infer the geometry, velocity and temperature of the emitting molecular layers.

Installation
-----------

You can install dynamite using pip:

.. code-block:: bash

    git clone https://github.com/cpinte/dynamite.git
    cd dynamite
    python3 setup.py install

If you don't have sudo rights, use:

.. code-block:: bash

    python3 setup.py install --user

To install in developer mode (using symlinks to point directly at this directory, so that code changes here are immediately available without needing to repeat the installation):

.. code-block:: bash

    python3 setup.py develop

Dependencies
-----------

The package requires the following Python packages:

- numpy
- astropy
- scipy
- matplotlib
- alive-progress

Main Features
------------

The package provides functionality for:

1. Extracting emission surfaces from molecular line data
2. Performing multi-scale analysis of the data
3. Computing disk geometry, velocity and temperature profiles
4. Fitting surface height profiles with various models
5. Estimating central stellar mass
6. Visualizing results with customizable plots

Core Components
--------------

Surface Class
~~~~~~~~~~~~

The main class for analyzing molecular line data is the `Surface` class. It provides methods for:

- Initializing and processing data cubes
- Extracting emission surfaces
- Computing disk parameters
- Fitting and analyzing surface profiles
- Plotting results

Key Methods
~~~~~~~~~~

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   surface
   examples

API Reference
------------

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/surface
   api/utils

Examples
--------

.. toctree::
   :maxdepth: 2
   :caption: Examples:

   examples/basic_usage
   examples/advanced_analysis

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 