#ifndef HMMBasicTypes_H
#define HMMBasicTypes_H

/*!
  \file HMMBasicTypes.hpp
  \brief Header file for Basic types (sequences, reals, etc.) used to implement Hidden Markov Models.
  \author DOP (dohmatob elvis dopgima)
*/

// Boost 
#include <boost/numeric/ublas/matrix.hpp> // matrix algebra
#include <boost/numeric/ublas/matrix_proxy.hpp> // extract matrix row, matrix column, etc.
#include <boost/numeric/ublas/io.hpp> // display matrix, etc.
#include <boost/multi_array.hpp> // for multi-dimensional arrays (aka tensors), etc.
#include <boost/assign/std/vector.hpp> // operator+=() for vectors, etc.
#include <boost/assert.hpp>
#include <boost/tuple/tuple.hpp> // so I can return multiple values from functions (like in python)

namespace ublas = boost::numeric::ublas; // namespace alias (saner than 'using ...')
using namespace boost::assign; 

namespace HiddenMarkovModels
{
  /*!
    \typedef real_type
    \brief Internal representation of real numbers .
  */
  typedef double real_type;   

  /*!
    \typedef sequence_type
    \brief Internal representation of sequences.
  */
  typedef std::vector<unsigned int> sequence_type;

  /*!
    \typedef matrix
    \brief Internal representation of probability matrices.
  */
  typedef ublas::matrix<real_type> matrix;

  /*!
    \typedef vector
    \brief Internal representation of probability vectors.
  */  
  typedef ublas::vector<real_type> vector;
}

#endif // HMMBasicTypes_H
