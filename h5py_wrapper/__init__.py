# encoding: utf8

"""
h5py_wrapper
============

Wrapper to conveniently store arbitrarily nested python dictionaries
to hdf5 files.

The dictionaries are stored in an hdf5 file by creating groups for
every level and a dataset for the value in the lowest level.

Functions
---------

- save : store nested dictionary in hdf5 file
- load : load nested dictionary from hdf5 file

"""

from .wrapper import save
from .wrapper import load


__version__ = '1.0.0'
