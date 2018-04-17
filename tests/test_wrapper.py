# -*- coding: utf-8 -*-
"""
Unit and integration tests for the h5py_wrapper module

"""

from future.builtins import str, range
import importlib
import os
import numpy as np
from numpy.testing import assert_array_equal
import pytest
import sys

import h5py_wrapper.wrapper as h5w
import h5py_wrapper.lib as h5w_lib

# check whether quantities is available
try:
    import quantities as pq
    quantities_found = True
except ImportError:
    quantities_found = False

fn = 'data.h5'
fn2 = 'data2.h5'

# define data
i0 = 6
f0 = 3.14159
s0 = 'this is a test'
b0 = True

l0i = [1, 2, 3, 4, 5]
l0s = ['a', 'b', 'c']
ll0 = [[6, 7, 8], [9, 10, 11]]
ln0 = [[12, 13], [14, 15, 16]]
lc0 = [1+1.j, 2+2.j, 3+3.j]

t0i = (1, 2, 3, 4, 5)
t0s = ('a', 'b', 'c')
tt0 = ((6, 7, 8), (9, 10, 11))
tn0 = ((1, 2), (3, 4, 5))

d0 = {'i': i0, 'f': f0, 's': s0}
dn0 = {'d1': d0, 'd2': d0}

# define containers
simpledata_str = ['i', 'f', 's', 'b', 'c']
simpledata_val = [i0, f0, s0, b0]

arraydata_str = ['ai', 'as', 'm', 'c']
arraydata_val = [np.array(l0i),
                 np.array(l0s),
                 np.array(ll0),
                 np.array(lc0)]

listdata_str = ['li', 'ls', 'm', 'ln']
listdata_val = [l0i, l0s, ll0, ln0]

dictdata_str = ['d']
dictdata_val = [d0]

tupledata_str = ['ti', 'tf', 'ts', 'tn']
tupledata_val = [t0i, t0s, tt0, tn0]


pytestmark = pytest.mark.usefixtures("cleanup")


def _construct_simpledata():
    res = {}
    for key, val in zip(simpledata_str, simpledata_val):
        res[key] = val
    return res


def test_write_and_load_with_label():
    res = _construct_simpledata()
    h5w.save(fn, res, write_mode='w', path='test_label')
    for key, val in zip(simpledata_str, simpledata_val):
        assert(h5w.load(fn, 'test_label/' + key) == val)


def test_store_and_load_dataset_directly():
    res = _construct_simpledata()
    h5w.save(fn, res, write_mode='w')
    for key, val in zip(simpledata_str, simpledata_val):
        assert(h5w.load(fn, '/' + key) == val)


def test_old_store_and_load_simpledata():
    res = _construct_simpledata()
    h5w.save(fn, res, write_mode='w')
    res.clear()
    res = h5w.load(fn)
    for key, val in zip(simpledata_str, simpledata_val):
        assert(res[key] == val)


def test_store_and_load_simpledata():
    res = _construct_simpledata()
    h5w.save(fn, res, write_mode='w')
    res.clear()
    res = h5w.load(fn)
    for key, val in zip(simpledata_str, simpledata_val):
        assert(res[key] == val)


def test_store_and_load_arraydata():
    res = {}
    for key, val in zip(arraydata_str, arraydata_val):
        res[key] = val
    h5w.save(fn, res, write_mode='w')
    res.clear()
    res = h5w.load(fn)
    for key, val in zip(arraydata_str, arraydata_val):
        assert_array_equal(res[key], val)


def test_store_and_load_listdata():
    res = {}
    for key, val in zip(listdata_str, listdata_val):
        res[key] = val
    h5w.save(fn, res, write_mode='w')
    res.clear()
    res = h5w.load(fn)
    for key, val in zip(listdata_str, listdata_val):
        if isinstance(val[0], list):
            for ii in range(len(val)):
                assert(isinstance(res[key][ii], list))
                assert_array_equal(res[key][ii], val[ii])
        else:
            assert(isinstance(res[key], type(val)))
            assert_array_equal(res[key], val)


def test_store_and_load_tupledata():
    res = {}
    for key, val in zip(tupledata_str, tupledata_val):
        res[key] = val
    h5w.save(fn, res, write_mode='w')
    res.clear()
    res = h5w.load(fn)
    for key, val in zip(tupledata_str, tupledata_val):
        assert(isinstance(res[key], tuple))
        assert_array_equal(res[key], np.array(val))


def test_store_and_load_dictdata():
    res = {}
    for key, val in zip(dictdata_str, dictdata_val):
        res[key] = val
    h5w.save(fn, res, write_mode='w')
    res.clear()
    res = h5w.load(fn)
    for dkey, dval in zip(dictdata_str, dictdata_val):
        for key, val in dval.items():
            assert(res[dkey][key] == val)


def test_store_and_load_numpy_datatypes():
    res = {}
    res['float64'] = np.float64(f0)
    res['int64'] = np.int64(i0)
    h5w.save(fn, res, write_mode='w')
    res.clear()
    res = h5w.load(fn)
    assert(isinstance(res['float64'], np.float64))
    assert(isinstance(res['int64'], np.int64))


def test_overwrite_dataset():
    res = {'a': 5}
    h5w.save(fn, res, write_mode='w')
    res.clear()
    res = {'a': 6}
    with pytest.raises(KeyError):
        h5w.save(fn, res, write_mode='a', overwrite_dataset=False)
    res.clear()
    res = h5w.load(fn)
    assert(res['a'] == 5)  # dataset should still contain old value
    res.clear()
    res = {'a': 6}
    h5w.save(
        fn, res, write_mode='a', overwrite_dataset=True)
    res.clear()
    res = h5w.load(fn)
    assert(res['a'] == 6)  # dataset should contain new value


def test_write_empty_array():
    res = {'a': [], 'b': np.array([])}
    h5w.save(fn, res, write_mode='w')
    res.clear()
    res = h5w.load(fn)
    assert_array_equal(res['a'], [])
    assert_array_equal(res['b'], [])


def test_write_nested_empty_array():
    res = {'a': [[], []], 'b': np.array([[], []])}
    h5w.save(fn, res, write_mode='w')
    res.clear()
    res = h5w.load(fn)
    assert_array_equal(res['a'], [[], []])
    assert(np.shape(res['a']) == (2, 0))
    assert_array_equal(res['b'], [[], []])
    assert(np.shape(res['b']) == (2, 0))


def test_read_empty_array_via_path():
    res = {'a': np.array([[], []])}
    h5w.save(fn, res, write_mode='w')
    res.clear()
    res = h5w.load(fn, path='a')
    assert_array_equal(res, [[], []])
    assert(np.shape(res) == (2, 0))


def test_handle_nonexisting_path():
    res = {}
    stest = 'this is a test'
    h5w.save(fn, res, write_mode='w')
    try:
        res = h5w.load(fn, path='test/')
        raise Exception()  # should not get until here
    except KeyError:
        res['test'] = stest
        h5w.save(fn, res)
        res.clear()
        res = h5w.load(fn, path='test/')
        assert(res == stest)


def test_store_none():
    res = {'a1': None}
    h5w.save(fn, res, write_mode='w')
    res.clear()
    res = h5w.load(fn)
    assert(res['a1'] is None)


def test_handle_nonexisting_file():
    try:
        h5w.load('asdasd.h5')
        raise Exception()  # should not get until here
    except IOError:
        pass


def test_store_and_load_custom_array():
    a = np.array([[1, 2, 3, 4], [6, 7]])
    h5w.save(fn, {'a': a}, overwrite_dataset=True)
    # loading the whole data
    res = h5w.load(fn)
    for i in range(len(a)):
        assert_array_equal(a[i], res['a'][i])
    # loading path directly
    res = h5w.load(fn, path='a/')
    for i in range(len(a)):
        assert_array_equal(a[i], res[i])


@pytest.mark.skipif(not quantities_found, reason='quantities module not found.')
def test_store_and_load_quantities_array():
    data = {'times': np.array([1, 2, 3]) * pq.ms, 'positions':
            np.array([1, 2, 3]) * pq.cm}
    h5w.save(fn, data, overwrite_dataset=True)
    # loading the whole data
    res = h5w.load(fn)
    assert(res['times'].dimensionality == data['times'].dimensionality)


def test_store_and_load_with_compression():
    data = {'a': 1, 'test1': {'b': 2}, 'test2': {
        'test3': {'c': np.array([1, 2, 3])}}}
    h5w.save(fn, data, write_mode='w', compression='gzip')
    h5w.load(fn)


def test_store_and_test_key_types():
    data = {'a': 1, (1, 2): {4: 2.}, 4.: 3.}
    h5w.save(fn, data, write_mode='w')
    res = h5w.load(fn)

    keys = ['a', (1, 2), 4.]
    for k in keys:
        assert(k in res)
    assert(4 in res[(1, 2)])


def test_load_lazy_simple():
    res = _construct_simpledata()
    h5w.save(fn, res, write_mode='w')
    res.clear()
    res = h5w.load(fn, lazy=True)
    for key, obj in res.items():
        assert(obj is None)


def test_load_lazy_nested():
    res = {'a': 1, 'test1': {'b': 2}, 'test2': {
        'test3': {'c': np.array([1, 2, 3])}}}
    h5w.save(fn, res, write_mode='w')
    res.clear()
    res = h5w.load(fn, lazy=True)
    assert(res['a'] is None)
    assert(res['test1']['b'] is None)
    assert(res['test2']['test3']['c'] is None)


def test_file_close_on_exception():
    res = {'a': 5}
    h5w.save(fn, res, write_mode='w')
    try:
        h5w.save(fn, res, write_mode='a', overwrite_dataset=False)
    except KeyError:
        pass
    h5w.save(fn, res, write_mode='w')


# This test is only run for Python 2 because version v0.0.1
# and 1.0.1 are not compatible with Python 3.
# Once there are releases compatible with Python 3, we have to
# change this.
@pytest.mark.skipif(sys.version_info >= (3, 0, 0),
                    reason='Previous release requires Python2')
def test_conversion_script(tmpdir):
    # convert LocalPath object to str to ensure that path can be
    # handled by os.path.join()
    path = str(tmpdir)

    previous_versions = ['0.0.1', '1.0.1']
    for version in previous_versions:
        print(version)
        version_stripped = version.replace('.', '')
        release_base_name = '_'.join(('h5py_wrapper', version_stripped))
        module = '.'.join((release_base_name, 'wrapper'))
        try:
            h5w_prev = importlib.import_module(module)
        except ImportError:
            h5w_lib.get_previous_version(version, tmpdir)
            sys.path.append(path)
            print(sys.path[-1])
            print(module)
            h5w_prev = importlib.import_module(module)

        tmp_fn = os.path.join(path, fn)
        tmp_fn2 = os.path.join(path, fn2)
        res = {key: value for key, value in zip(simpledata_str, simpledata_val)}
        res.update({key: value for key, value in zip(arraydata_str, arraydata_val)})
        h5w_prev.add_to_h5(tmp_fn, res)
        h5w_prev.add_to_h5(tmp_fn2, res)
        with open('conversion_list.txt', 'w') as f:
            f.write(tmp_fn)
            f.write('\n')
            f.write(tmp_fn2)

        # Specify list on command line
        conversion_file_cmd = 'convert_h5file'
        cmd = 'python {}/../{} {} {} --release={}'.format(os.path.dirname(__file__),
                                                          conversion_file_cmd,
                                                          tmp_fn, tmp_fn2,
                                                          version)
        os.system(cmd)
        # Read list of files from file and pipe into conversion script
        cmd = 'cat conversion_list.txt | python {}/../{} {}={}'.format(os.path.dirname(__file__),
                                                                       conversion_file_cmd,
                                                                       '--release',
                                                                       version)
        os.system(cmd)
        os.remove('conversion_list.txt')
        # Find files based on pattern using `find` and pipe into conversion script
        cmd = 'find -name "data*.h5" | python {}/../{} {}={}'.format(os.path.dirname(__file__),
                                                                     conversion_file_cmd,
                                                                     '--release',
                                                                     version)
        os.system(cmd)

        res2 = h5w.load(tmp_fn)
        for key, value in res.items():
            if isinstance(res2[key], np.ndarray):
                assert_array_equal(res2[key], value)
            else:
                assert(res2[key] == value)
            if isinstance(res2[key], str):
                assert(isinstance(res2[key], type(str(value))))
            else:
                assert(isinstance(res2[key], type(value)))
        os.remove(tmp_fn)
        os.remove(tmp_fn2)


def test_raises_error_for_dictlabel_and_path():
    res = {}
    with pytest.raises(ValueError):
        h5w.save(fn, res, dict_label='test', path='test')


@pytest.fixture()
def cleanup():
    yield
    try:
        os.remove(fn)
    except OSError:
        pass

    try:
        os.remove(fn2)
    except OSError:
        pass
