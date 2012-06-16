#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp> // matrix row, matrix colum, etc.
#include <boost/numeric/ublas/io.hpp> // display matrix, etc.
#include <boost/multi_array.hpp> // for multi-dimensional arrays (aka tensors), etc.
#include <boost/assign/std/vector.hpp> // operator+=() for vectors, etc.
#include <boost/assert.hpp>
#include <boost/tuple/tuple.hpp> // so I can return multiple values from functions (like in python)
#include <math.h> // log, abs, etc.
#include <iostream> // cout, etc.
#include <fstream> // file handling functions like getline, etc.
#include <sstream>

using namespace boost::numeric::ublas;
using namespace boost::assign;

typedef double real_type;
typedef std::vector<unsigned int> sequence_type;
