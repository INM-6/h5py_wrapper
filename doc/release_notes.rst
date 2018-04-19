====================
Release notes
====================

.. toctree::
   :maxdepth: 2

1.1.0
=====

https://github.com/INM-6/h5py_wrapper/releases/tag/1.1.0

This release adds Python3 support to the wrapper. This transition required major changes to the code base, particularly the handling of strings in the wrapper was revised.

Furthermore, we added:

- support for complex numbers
- support for numpy 64bit data types.
- improved the error handling and warnings.

This release contains 48 commits, 11 closed issues and 18 merged pull-requests.

API changes
+++++++++++

The deprecated functions `add_to_h5` and `load_h5` were removed.

Contributors
++++++++++++

- Maximilian Schmidt
- Jakob Jordan
- Julia Sprenger

1.0.1
=====

https://github.com/INM-6/h5py_wrapper/releases/tag/1.0.1

API changes
+++++++++++

The old function names `load_h5` and `add_to_h5` have been replaced by load and save.
The old function names are still available, but will raise a DeprecationWarning and be removed in one of the next releases.
New features and functionality

The wrapper now explicitly stores the types of values of the dictionary in the hdf5 file and is able to retrieve the original value types. A list of supported value types is provided in the documentation.

This has the effect that old files created with release version 0.0.1 cannot be loaded with 1.0.1. To remedy this, we provide a conversion script that converts old files to the new format. The user can provide

    - single file names,
    - lists of file names
    - files containing file names
    - input from stdin.

Contributors
++++++++++++

- Maximilian Schmidt
- Jakob Jordan
	      
v0.0.1
======

https://github.com/INM-6/h5py_wrapper/releases/tag/v0.0.1

This release contains 16 commits since the migration to github.
The wrapper provides the two basic functions `wrapper.add_to_h5` and `wrapper.load_h5`.
Furthermore, the user can load h5 files (load_h5_old) that were created with a deprecated version of the wrapper and transform these files into the current format (transform_h5).


Contributors
++++++++++++

(in alphabetical order)

- Hannah Bos
- Michael Denker
- Jakob Jordan
- Maximilian Schmidt
- Jannis Schuecker
