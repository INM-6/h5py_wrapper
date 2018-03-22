# encoding: utf8
"""
Collection of convenience functions.

"""

import numpy as np
import os
import requests
import tarfile


def get_previous_version(version, path):
    """
    Retrieves the given version of the wrapper from github as a tar
    archive and extracts its contents to the current directory.

    Parameters
    ----------
    version : str
        Version number in format 'X.X.X'
    path : str
        Path to store the files.

    Returns
    -------
    package_dir : str
        Path to package.
    """
    base_url = "https://github.com/INM-6/h5py_wrapper/archive/"
    if version == '0.0.1':
        ver = 'v0.0.1'
    else:
        ver = version
    r = requests.get(''.join((base_url, ver, ".tar.gz")))
    # convert LocalPath object to str (if path to tmp dir is passed by py.test)
    # to ensure that path can be handled by os.path.join()
    path = str(path)
    try:
        r.raise_for_status()
        fn = ''.join((os.path.join(path, version), '.tar.gz'))
        with open(fn, 'wb') as f:
            f.write(r.content)
        with tarfile.open(fn) as f:
            f.extractall(path=path)

        # This case distinction is necessary because the directory structure
        # changes between versions
        if version == '0.0.1':
            pkg_wrp_dir = os.path.join(path, '-'.join(('h5py_wrapper', version)))
        elif version == '1.0.1':
            pkg_wrp_dir = os.path.join(path, '-'.join(('h5py_wrapper', version)), 'h5py_wrapper')
        package_dir = os.path.join(path, '_'.join(('h5py_wrapper', version.replace('.', ''))))
        os.rename(pkg_wrp_dir, package_dir)
        return package_dir
    except requests.exceptions.HTTPError:
        raise ImportError("Requested release version does not exist.")


def accumulate(iterator):
    """
    Creates a generator to iterate over the accumulated
    values of the given iterator.
    """
    total = 0
    for item in iterator:
        yield total, item
        total += item


def convert_numpy_types_in_dict(d):
    """
    Convert all numpy datatypes to default datatypes in a dictionary (in place).
    """
    for key, value in d.items():
        if isinstance(value, dict):
            convert_numpy_types_in_dict(value)
        elif isinstance(value, (np.float)):
            d[key] = float(value)
        elif isinstance(value, (np.bool, np.bool_)):
            d[key] = bool(value)
        elif isinstance(value, (np.int)):
            d[key] = int(value)


def convert_iterable_to_numpy_array(it):
    """
    Converts an iterable to a numpy array. If the elements of the
    iterable are strings, numpy unicode types are avoided by changing
    dtype to np.string_ to ensure h5py compatibility. See
    http://docs.h5py.org/en/latest/strings.html#what-about-numpy-s-u-type.
    """
    array = np.array(it)
    if array.dtype.kind == 'U':
        return array.astype(np.string_)
    else:
        return array
