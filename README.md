# dynamite

DYNamical Analysis and MultIscale Tomography of line Emission

dynamite is a python package that perform kinematic and tomographic analysis of sub-mm molecular data of protoplanetary disks. 
It follows the method presented in Pinte et al. 2018, with various improvements, to infer the geometry, velocity and temperature of the emitting molecular layers.


## Installation:

```
git clone https://github.com/cpinte/dynamite.git
cd dynamite
python3 setup.py install
```

If you don't have the `sudo` rights, use `python3 setup.py install --user`.

To install in developer mode: (i.e. using symlinks to point directly
at this directory, so that code changes here are immediately available
without needing to repeat the above step):

```
 python3 setup.py develop
```
