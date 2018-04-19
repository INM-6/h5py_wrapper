# encoding: utf8
from setuptools import setup

setup(
    name='h5py_wrapper',
    version='1.1.0',
    author='Maximilian Schmidt, Jakob Jordan',
    author_email='max.schmidt@fz-juelich.de',
    description=('A wrapper to conveniently store nested Python dictionaries in hdf5 files.'),
    license='GPL2',
    keywords='hdf5 h5py',
    url='https://github.com/INM-6/h5py_wrapper',
    packages=['h5py_wrapper', 'tests'],
    python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*, !=3.4.*, <4',
    scripts=['convert_h5file'],
    install_requires=['h5py', 'pytest-runner', 'future'],
    tests_require=['pytest'],
    long_description=open('README.rst').read(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Utilities',
    ],
)
