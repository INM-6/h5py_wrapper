##########################################
## Test script for h5py_wrapper.wrapper ##
##########################################

import h5py_wrapper.wrapper as h5w

import numpy
from numpy.testing import assert_array_equal

fn = 'data.h5'

# define data
i0 = 6
f0 = 3.14159
s0 = 'this is a test'

a0i = [1,2,3,4,5]
a0s = ['a','b','c']
m0 = [[6,7,8],[9,10,11]]
#an0 = [[12,13],[14,15,16]] #NESTED ARRAY FAILS DUE TO UNKOWN OBJECT TYPE

d0 = {'i':i0,'f':f0,'s':s0}
dn0= {'d1':d0,'d2':d0}

# define containers
simpledata_str = ['i','f','s']
simpledata_val = [i0,f0,s0]

arraydata_str = ['ai','as','m']
arraydata_val = [a0i,a0s,m0]

dictdata_str = ['d']
dictdata_val = [d0]


def store_and_load_dataset_directly(label=''):
    res = {}
    for key,val in zip(simpledata_str,simpledata_val):
        res[key] = val
    h5w.add_to_h5(fn,res,write_mode='w',dict_label=label)
    for key,val in zip(simpledata_str,simpledata_val):
        assert(h5w.load_h5(fn,label+'/'+key) == val)

def old_store_and_load_simpledata(label=''):
    res = {}
    for key,val in zip(simpledata_str,simpledata_val):
        res[key] = val
    h5w.add_to_h5(fn,res,write_mode='w',dict_label=label)
    res.clear()
    if label == '':
        res = h5w.load_h5(fn)
    else:
        res = h5w.load_h5(fn)[label]
    for key,val in zip(simpledata_str,simpledata_val):
        assert(res[key] == val)

def store_and_load_simpledata(label=''):
    res = {}
    for key,val in zip(simpledata_str,simpledata_val):
        res[key] = val
    h5w.add_to_h5(fn,res,write_mode='w',dict_label=label)
    res.clear()
    res = h5w.load_h5(fn,label)
    for key,val in zip(simpledata_str,simpledata_val):
        assert(res[key] == val)

def store_and_load_arraydata(label=''):
    res = {}
    for key,val in zip(arraydata_str,arraydata_val):
        res[key] = val
    h5w.add_to_h5(fn,res,write_mode='w',dict_label=label)
    res.clear()
    res = h5w.load_h5(fn,label)
    for key,val in zip(arraydata_str,arraydata_val):
        assert_array_equal(res[key],val)

def store_and_load_dictdata(label=''):
    res = {}
    for key,val in zip(dictdata_str,dictdata_val):
        res[key] = val
    h5w.add_to_h5(fn,res,write_mode='w',dict_label=label)
    res.clear()
    res = h5w.load_h5(fn,label)
    for dkey,dval in zip(dictdata_str,dictdata_val):
        for key,val in dval.items():
            assert(res[dkey][key] == val)

def check_for_node(label=''):
    res = {'a':1,'test1':{'b':2},'test2':{'test3':{'c':3}}}
    h5w.add_to_h5(fn,res,write_mode='w',dict_label=label)
    if label != '':
        assert(h5w.node_exists(fn,label) == True)
    assert(h5w.node_exists(fn,label+'/a') == True)
    assert(h5w.node_exists(fn,label+'/nota') == False)
    assert(h5w.node_exists(fn,label+'/test1/b') == True)
    assert(h5w.node_exists(fn,label+'/test1/notb') == False)
    assert(h5w.node_exists(fn,label+'/test2/test3/c') == True)
    assert(h5w.node_exists(fn,label+'/test2/test3/notc') == False)

def overwrite_dataset(label=''):
    res = {'a':5}
    h5w.add_to_h5(fn,res,write_mode='w',dict_label=label)
    res.clear()
    res = {'a':6}
    h5w.add_to_h5(fn,res,write_mode='a',overwrite_dataset=False,dict_label=label)
    res.clear()
    res = h5w.load_h5(fn,label)
    assert(res['a'] == 5) # dataset should still contain old value
    res.clear()
    res = {'a':6}
    h5w.add_to_h5(fn,res,write_mode='a',overwrite_dataset=True,dict_label=label)
    res.clear()
    res = h5w.load_h5(fn,label)
    assert(res['a'] == 6) # dataset should contain new value

def write_empty_array(label=''):
    res = {'a':[],'b':numpy.array([])}
    h5w.add_to_h5(fn,res,write_mode='w',dict_label=label)
    res.clear()
    res = h5w.load_h5(fn,label)
    assert_array_equal(res['a'],[])
    assert_array_equal(res['b'],[])

def write_nested_empty_array(label=''):
    res = {'a':[[],[]],'b':numpy.array([[],[]])}
    h5w.add_to_h5(fn,res,write_mode='w',dict_label=label)
    res.clear()
    res = h5w.load_h5(fn,label)
    assert_array_equal(res['a'],[[],[]])
    assert(numpy.shape(res['a']) == (2,0))
    assert_array_equal(res['b'],[[],[]])
    assert(numpy.shape(res['b']) == (2,0))

def read_empty_array_via_path():
    res = {'a': numpy.array([[],[]])}
    h5w.add_to_h5(fn, res, dict_label='test_label')
    res.clear()
    res = h5w.load_h5(fn, path='test_label/a')
    assert_array_equal(res, [[],[]])
    assert(numpy.shape(res) == (2,0))

def handle_nonexisting_path(label=''):
    res = {}
    stest = 'this is a test'
    h5w.add_to_h5(fn,res,write_mode='w',dict_label=label)
    try:
        res = h5w.load_h5(fn,path='test/')
        raise Exception() # should not get until here
    except KeyError:
        res['test'] = stest
        h5w.add_to_h5(fn,res)
        res.clear()
        res = h5w.load_h5(fn,path='test/')
        assert(res == stest)

def store_none():
    res = {'a1':None}
    h5w.add_to_h5(fn,res,write_mode='w')
    res.clear()
    res = h5w.load_h5(fn)
    assert(res['a1'] == None)

def handle_nonexisting_file():
    try:
        res = h5w.load_h5('asdasd.h5')
        raise Exception() # should not get until here
    except IOError:
        pass

def store_and_load_custom_array():
    a = [[1,2,3,4],[6,7]]
    h5w.add_to_h5(fn,{'a': a},overwrite_dataset=True)
    # loading the whole data
    res = h5w.load_h5(fn)
    for i in xrange(len(a)):
        assert(numpy.sum(a[i]-res['a'][i]) < 1e-12)
    # loading path directly
    res = h5w.load_h5(fn, path='a/')
    for i in xrange(len(a)):
        assert(numpy.sum(a[i]-res[i]) < 1e-12)

def store_and_load_quantities_array() :
    import quantities as pq
    data = {'times' : numpy.array([1,2,3])*pq.ms, 'positions' : numpy.array([1,2,3])*pq.cm}
    h5w.add_to_h5(fn, data, overwrite_dataset=True)
    # loading the whole data
    res = h5w.load_h5(fn)
    assert(res['times'].dimensionality == data['times'].dimensionality)


def store_and_load_with_compression() :
    data = {'a':1,'test1':{'b':2},'test2':{'test3':{'c':numpy.array([1,2,3])}}}
    h5w.add_to_h5(fn,data,write_mode='w', compression='gzip')
    res = h5w.load_h5(fn)

def store_and_test_key_types() :
    data = {'a' : 1, (1,2) : 2., 4. : 3.}
    h5w.add_to_h5(fn,data,write_mode='w', compression='gzip')
    res = h5w.load_h5(fn)

    keys = ['a',(1,2),4.]
    for k in keys :
        assert(k in res.keys())
#def handle_existing_dataset
#def handle_existing_group

############################################################################
# Start tests

# simple data: integer, float, string
# array data: non-nested array, matrix
# dict data: non-nested dictionary containing simple data
# nested dict: nested dictionary containing dict data

# test storing and loading of:

# simple data without given dict_label, loading datasets directly
store_and_load_dataset_directly()
# simple data with given dict_label, loading datasets directly
store_and_load_dataset_directly('testlabel')

# simple data with given dict_label, not using path variable for label
old_store_and_load_simpledata('testlabel')

# simple data without given dict_label
store_and_load_simpledata()
# simple data with given dict_label
store_and_load_simpledata('testlabel')

# array data without given dict_label
store_and_load_arraydata()
# array data with given dict_label
store_and_load_arraydata('test_label')

# dict data without given dict_label
store_and_load_dictdata()
# dict_data with diven dict_label
store_and_load_dictdata('testlabel')

# check whether a dataset exists
check_for_node()
check_for_node('testlabel')

# test overwriting a dataset
overwrite_dataset()
overwrite_dataset('testlabel')

# test writing empty array
write_empty_array()
write_empty_array('testlabel')
read_empty_array_via_path()

# test writing empty nested array
write_nested_empty_array()
write_nested_empty_array('testlabel')

# test loading of nonexisting path
handle_nonexisting_path()
handle_nonexisting_path('testlabel')

# test storing None
store_none()

# test opening nonexistent file
handle_nonexisting_file()

# test array with unequal lengths of entries
store_and_load_custom_array()

# test storing and loading of quantities.Quantity
store_and_load_quantities_array()

# test compression of data
store_and_load_with_compression()

# test storage of different types of keys
store_and_test_key_types()

print 'test_wrapper: success'
