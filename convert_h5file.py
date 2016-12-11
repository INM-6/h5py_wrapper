#!/usr/bin/env python
# encoding: utf8

'''
Conversion script to convert files from release
version 0.0.1 to 1.0.

Usage: convert_h5file [-h|--help] <filename> [--save-backup] [-v|--verbose]

Options:
    --save-backup   save backup version of file in old file format [default: False].
    -v, --verbose  print informative output to screen
    -h, --help      print this text
'''

import docopt
import wrapper as h5w
import urllib
import tarfile
import numpy as np
import os

if __name__ == '__main__':
    def dict_check_for_numpy_types(d):
        '''
        Convert all numpy datatypes to default datatypes in a dictionary.
        '''
        for key, value in d.items():
            if isinstance(value, dict):
                dict_check_for_numpy_types(value)
            elif isinstance(value, (np.int)):
                d[key] = int(value)
            elif isinstance(value, (np.float)):
                d[key] = float(value)

    # First get 0.0.1 release version
    try:
        import h5py_wrapper_001.wrapper as h5w_001
    except ImportError:
        urllib.urlretrieve("https://github.com/INM-6/h5py_wrapper/archive/v0.0.1.tar.gz",
                           filename='./v0.0.1.tar.gz')
        f = tarfile.open('v0.0.1.tar.gz')
        f.extract('h5py_wrapper-0.0.1/wrapper.py')
        f.extract('h5py_wrapper-0.0.1/__init__.py')
        os.rename('h5py_wrapper-0.0.1', 'h5py_wrapper_001')
        import h5py_wrapper_001.wrapper as h5w_001

    args = docopt.docopt(__doc__)
    if args['--verbose']:
        print("Loading file.")
    d = h5w_001.load_h5(args['<filename>'])

    # This step is necessary because the 0.0.1 release loads int and float types as numpy datatypes,
    # which are not supported as scalar datatypes by the 1.0 release version.
    if args['--verbose']:
        print("Checking for numpy datatypes in scalar values.")

    dict_check_for_numpy_types(d)

    if args['--save-backup']:
        if args['--verbose']:
            print("Saving backup file.")
        backup_name = args['<filename>'].split('.')[0] + '_001.' + args['<filename>'].split('.')[1]
        os.rename(args['<filename>'], backup_name)

    if args['--verbose']:
        print("Saving file in new format.")
    h5w.save(args['<filename>'], d, write_mode='w')
