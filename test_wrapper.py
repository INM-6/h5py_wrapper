# -*- coding: utf-8 -*-
"""
Unit and integration tests for the h5py_wrapper module

"""

import unittest
import wrapper as h5w
import numpy as np
from numpy.testing import assert_array_equal

# check whether quantities is available
try:
    import quantities as pq
    quantities_found = True
except ImportError:
    quantities_found = False

fn = 'data.h5'

# define data
i0 = 6
f0 = 3.14159
s0 = 'this is a test'

l0i = [1, 2, 3, 4, 5]
l0s = ['a', 'b', 'c']
ll0 = [[6, 7, 8], [9, 10, 11]]
# ln0 = [[12,13],[14,15,16]] #NESTED ARRAY FAILS DUE TO UNKOWN OBJECT TYPE

t0i = (1, 2, 3, 4, 5)
t0s = ('a', 'b', 'c')
tt0 = ((6, 7, 8), (9, 10, 11))


d0 = {'i': i0, 'f': f0, 's': s0}
dn0 = {'d1': d0, 'd2': d0}

# define containers
simpledata_str = ['i', 'f', 's']
simpledata_val = [i0, f0, s0]

arraydata_str = ['ai', 'as', 'm']
arraydata_val = [np.array(l0i), np.array(l0s), np.array(ll0)]

listdata_str = ['li', 'ls', 'm']
listdata_val = [l0i, l0s, ll0]

dictdata_str = ['d']
dictdata_val = [d0]

tupledata_str = ['ti', 'tf', 'ts']
tupledata_val = [t0i, t0s, tt0]


class WrapperTest(unittest.TestCase):
    def construct_simpledata(self):
        res = {}
        for key, val in zip(simpledata_str, simpledata_val):
            res[key] = val
        return res

    def test_write_and_load_with_label(self):
        res = self.construct_simpledata()
        h5w.add_to_h5(fn, res, write_mode='w', dict_label='test_label')
        for key, val in zip(simpledata_str, simpledata_val):
            self.assertEqual(h5w.load_h5(fn, 'test_label/' + key), val)

    def test_store_and_load_dataset_directly(self):
        res = self.construct_simpledata()
        h5w.add_to_h5(fn, res, write_mode='w')
        for key, val in zip(simpledata_str, simpledata_val):
            self.assertEqual(h5w.load_h5(fn, '/' + key), val)

    def test_old_store_and_load_simpledata(self):
        res = self.construct_simpledata()
        h5w.add_to_h5(fn, res, write_mode='w')
        res.clear()
        res = h5w.load_h5(fn)
        for key, val in zip(simpledata_str, simpledata_val):
            self.assertEqual(res[key], val)

    def test_store_and_load_simpledata(self):
        res = self.construct_simpledata()
        h5w.add_to_h5(fn, res, write_mode='w')
        res.clear()
        res = h5w.load_h5(fn)
        for key, val in zip(simpledata_str, simpledata_val):
            self.assertEqual(res[key], val)

    def test_store_and_load_arraydata(self):
        res = {}
        for key, val in zip(arraydata_str, arraydata_val):
            res[key] = val
        h5w.add_to_h5(fn, res, write_mode='w')
        res.clear()
        res = h5w.load_h5(fn)
        for key, val in zip(arraydata_str, arraydata_val):
            assert_array_equal(res[key], val)

    def test_store_and_load_listdata(self):
        res = {}
        for key, val in zip(listdata_str, listdata_val):
            res[key] = val
        h5w.add_to_h5(fn, res, write_mode='w')
        res.clear()
        res = h5w.load_h5(fn)
        for key, val in zip(listdata_str, listdata_val):
            assert_array_equal(res[key], val)

    def test_store_and_load_tupledata(self):
        res = {}
        for key, val in zip(tupledata_str, tupledata_val):
            res[key] = val
        h5w.add_to_h5(fn, res, write_mode='w')
        res.clear()
        res = h5w.load_h5(fn)
        for key, val in zip(tupledata_str, tupledata_val):
            assert_array_equal(res[key], np.array(val))

    def test_store_and_load_dictdata(self):
        res = {}
        for key, val in zip(dictdata_str, dictdata_val):
            res[key] = val
        h5w.add_to_h5(fn, res, write_mode='w')
        res.clear()
        res = h5w.load_h5(fn)
        for dkey, dval in zip(dictdata_str, dictdata_val):
            for key, val in dval.items():
                self.assertEqual(res[dkey][key], val)

    def test_overwrite_dataset(self):
        res = {'a': 5}
        h5w.add_to_h5(fn, res, write_mode='w')
        res.clear()
        res = {'a': 6}
        self.assertRaises(KeyError, h5w.add_to_h5,
                          fn, res, write_mode='a', overwrite_dataset=False)
        res.clear()
        res = h5w.load_h5(fn)
        self.assertEqual(res['a'], 5)  # dataset should still contain old value
        res.clear()
        res = {'a': 6}
        h5w.add_to_h5(
            fn, res, write_mode='a', overwrite_dataset=True)
        res.clear()
        res = h5w.load_h5(fn)
        self.assertEqual(res['a'], 6)  # dataset should contain new value

    def test_write_empty_array(self):
        res = {'a': [], 'b': np.array([])}
        h5w.add_to_h5(fn, res, write_mode='w')
        res.clear()
        res = h5w.load_h5(fn)
        assert_array_equal(res['a'], [])
        assert_array_equal(res['b'], [])

    def test_write_nested_empty_array(self):
        res = {'a': [[], []], 'b': np.array([[], []])}
        h5w.add_to_h5(fn, res, write_mode='w')
        res.clear()
        res = h5w.load_h5(fn)
        assert_array_equal(res['a'], [[], []])
        self.assertEqual(np.shape(res['a']), (2, 0))
        assert_array_equal(res['b'], [[], []])
        self.assertEqual(np.shape(res['b']), (2, 0))

    def test_read_empty_array_via_path(self):
        res = {'a': np.array([[], []])}
        h5w.add_to_h5(fn, res, write_mode='w')
        res.clear()
        res = h5w.load_h5(fn, path='a')
        assert_array_equal(res, [[], []])
        self.assertEqual(np.shape(res), (2, 0))

    def test_handle_nonexisting_path(self):
        res = {}
        stest = 'this is a test'
        h5w.add_to_h5(fn, res, write_mode='w')
        try:
            res = h5w.load_h5(fn, path='test/')
            raise Exception()  # should not get until here
        except KeyError:
            res['test'] = stest
            h5w.add_to_h5(fn, res)
            res.clear()
            res = h5w.load_h5(fn, path='test/')
            self.assertEqual(res, stest)

    def test_store_none(self):
        res = {'a1': None}
        h5w.add_to_h5(fn, res, write_mode='w')
        res.clear()
        res = h5w.load_h5(fn)
        self.assertIsNone(res['a1'])

    def test_handle_nonexisting_file(self):
        try:
            h5w.load_h5('asdasd.h5')
            raise Exception()  # should not get until here
        except IOError:
            pass

    def test_store_and_load_custom_array(self):
        a = [[1, 2, 3, 4], [6, 7]]
        h5w.add_to_h5(fn, {'a': a}, overwrite_dataset=True)
        # loading the whole data
        res = h5w.load_h5(fn)
        for i in xrange(len(a)):
            self.assertLess(np.sum(a[i] - res['a'][i]), 1e-12)
        # loading path directly
        res = h5w.load_h5(fn, path='a/')
        for i in xrange(len(a)):
            self.assertLess(np.sum(a[i] - res[i]), 1e-12)

    @unittest.skipUnless(quantities_found, 'No h5py_wrapper found.')
    def test_store_and_load_quantities_array(self):
        data = {'times': np.array([1, 2, 3]) * pq.ms, 'positions':
                np.array([1, 2, 3]) * pq.cm}
        h5w.add_to_h5(fn, data, overwrite_dataset=True)
        # loading the whole data
        res = h5w.load_h5(fn)
        self.assertEqual(res['times'].dimensionality,
                         data['times'].dimensionality)

    def test_store_and_load_with_compression(self):
        data = {'a': 1, 'test1': {'b': 2}, 'test2': {
            'test3': {'c': np.array([1, 2, 3])}}}
        h5w.add_to_h5(fn, data, write_mode='w', compression='gzip')
        h5w.load_h5(fn)

    def test_store_and_test_key_types(self):
        data = {'a': 1, (1, 2): {4: 2.}, 4.: 3.}
        h5w.add_to_h5(fn, data, write_mode='w')
        res = h5w.load_h5(fn)

        keys = ['a', (1, 2), 4.]
        for k in keys:
            self.assertIn(k, res.keys())
        self.assertIn(4, res[(1, 2)].keys())

    def test_load_lazy_simple(self):
        res = self.construct_simpledata()
        h5w.add_to_h5(fn, res, write_mode='w')
        res.clear()
        res = h5w.load_h5(fn, lazy=True)
        for key, obj in res.items():
            self.assertIsNone(obj)

    def test_load_lazy_nested(self):
        res = {'a': 1, 'test1': {'b': 2}, 'test2': {
            'test3': {'c': np.array([1, 2, 3])}}}
        h5w.add_to_h5(fn, res, write_mode='w')
        res.clear()
        res = h5w.load_h5(fn, lazy=True)
        self.assertIsNone(res['a'])
        self.assertIsNone(res['test1']['b'])
        self.assertIsNone(res['test2']['test3']['c'])
