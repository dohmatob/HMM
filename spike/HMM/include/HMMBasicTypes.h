#ifndef HMMBASICTYPES_H
#define HMMBASICTYPES_H

/*!
  \file HMMBasicTypes.h
  \brief Declaration of basic types used internally in Hidden Markov Models
  \author DOP (dohmatob elvis dopgima)
*/

#include <boost/numeric/ublas/matrix.hpp> // for ublas matrices and columns
#include <boost/numeric/ublas/matrix_proxy.hpp> // so we may create matrix proxies like rows and columns
#include <boost/multi_array.hpp>
#include <boost/numeric/ublas/io.hpp> // so we may display matrices, vectors, etc.
#include <vector>
#include <boost/tuple/tuple.hpp>
#include <fstream>
#include <iostream> // for std::ostream, etc.

/*!
  \namespace HiddenMarkovModels
  \brief Namespace for Hidden Markov Models.
*/
namespace HiddenMarkovModels
{
  /*!
    \typedef RealType
    \brief Internal representation of real numbers.
  */
  typedef double RealType;

  /*!
    \typedef RealVectorType
    \brief Internal reprensentation of real vectors (vectors with real-valued components).
  */
  typedef boost::numeric::ublas::vector< RealType > RealVectorType;

  /*!
    \typedef RealMatrixType
    \brief Internal reprensentation of real matrixs (matrices with real-valued entries).
  */
  typedef boost::numeric::ublas::matrix< RealType > RealMatrixType;

  /*!
    \typedef ObservationSequenceType
    \brief Internal representation of observation sequences.
  */
  typedef std::vector< int > ObservationSequenceType;
};

#endif // HMMBASICTYPES_H
