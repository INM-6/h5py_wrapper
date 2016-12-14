#!/usr/bin/env python
# encoding: utf8

"""
Conversion script to convert files from release
version 0.0.1 to 1.0.

Usage: convert_h5file [-h|--help] [<files>...] [--save-backup] [-v|--verbose] [--release=<version>]

Options:
    <files>  List of files to be converted, can be a file pattern
    --save-backup  save backup version of file in old file format [default: False].
    --release=<version>   release version used to create the file to be converted [default: 0.0.1]
    -v, --verbose  print informative output to screen
    -h, --help     print this text
"""

import docopt
import wrapper as h5w
import requests
import tarfile
import numpy as np
import os
import importlib
import sys


def _dict_check_for_numpy_types(d):
    """
    Convert all numpy datatypes to default datatypes in a dictionary.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            _dict_check_for_numpy_types(value)
        elif isinstance(value, (np.int)):
            d[key] = int(value)
        elif isinstance(value, (np.float)):
            d[key] = float(value)


def _get_previous_version(version):
    base_url = "https://github.com/INM-6/h5py_wrapper/archive/v"
    r = requests.get(''.join((base_url, version, ".tar.gz")))
    try:
        r.raise_for_status()
        fn = ''.join((os.path.join(os.getcwd(), version), '.tar.gz'))
        with open(fn, 'wb') as f:
            f.write(r.content)
        with tarfile.open(fn) as f:
            f.extract(''.join(('h5py_wrapper-', version, '/wrapper.py')))
            f.extract(''.join(('h5py_wrapper-', version, '/__init__.py')))
        os.rename('-'.join(('h5py_wrapper', version)),
                  '_'.join(('h5py_wrapper', version.replace('.', ''))))
    except requests.exceptions.HTTPError:
        raise ImportError("Requested release version does not exist.")

if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    # First get release version used to create the file to be converted
    version_stripped = args['--release'].replace('.', '')
    release_base_name = '_'.join(('h5py_wrapper', version_stripped))
    try:
        h5w_old = importlib.import_module('.'.join((release_base_name, 'wrapper')))
    except ImportError:
        _get_previous_version(args['--release'])
        h5w_old = importlib.import_module('.'.join((release_base_name, 'wrapper')))

    files = args['<files>'] or sys.stdin.read().splitlines()
    for fn in files:
        if args['--verbose']:
            print("Loading %s" % fn)
        d = h5w_old.load_h5(fn)

        # This step is necessary because the 0.0.1 release loads int and float types as numpy
        # datatypes, which are not supported as scalar datatypes by the 1.0 release version.
        if args['--verbose']:
            print("Checking for numpy datatypes in scalar values.")

        _dict_check_for_numpy_types(d)

        if args['--save-backup']:
            if args['--verbose']:
                print("Saving backup file.")
            orig_name = os.path.splitext(fn)
            backup_name = '.'.join((''.join((orig_name[0], version_stripped)), orig_name[1]))
            os.rename(fn, backup_name)

        if args['--verbose']:
            print("Saving file in new format.")
        h5w.save(fn, d, write_mode='w')
