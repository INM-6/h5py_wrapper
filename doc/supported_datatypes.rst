====================
Supported data types
====================

.. toctree::
   :maxdepth: 2


The wrapper stores the original data types of values of the dictionary in the produced hdf5 file.
The following data types are supported:

* float
    
* int
    
* str

* tuple

* numpy.array

* numpy.int64 and numpy.float64
    
* list
    
* bool
    
* quantities.Quantity (see https://pypi.python.org/pypi/quantities)
  
* Lists, tuples and numpy.arrays up to arbitrary depths if all dimensions are uniform, e.g.
  
  .. code-block:: python
		  
		  l = numpy.ones((3,3,3))

* Lists and tuples are required to contain the contain equal data types across one dimension. For instance, this raises an error

		    
  .. code-block:: python
		
		l = [[1,2], 'a']


			
* Lists, tuples and numpy.arrays with unequal dimensions with maximal depth 1.

  .. code-block:: python
		
		l = [[1,2], [1]]
