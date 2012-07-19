#ifndef HMMUTILS_H
#define HMMUTILS_H

/*!
  \file HMMUtils.h
  \brief Header file for routine function.
  \author DOP (dohmatob elvis dopgima)
*/

#include "HMMIOUtils.h"
#include "DiscreteHMM.h"

namespace HiddenMarkovModels
{
  /*!
    Checks a vector for stochasticity (i.e. all coordinates are nonnegative and sum is nonzero).

    \param v input vector
    \return true if vector is indeen stochastic, false otherwise
  */
  bool is_stochastic_vector(const HiddenMarkovModels::RealVectorType& v);

  /*!
    Checks a matrix for stochasticiy (i.e. all rows are stochastic).

    \param m input matrix
    \return true if matrix is indeen stochastic, false otherwise
  */
  bool is_stochastic_matrix(const HiddenMarkovModels::RealMatrixType& m);

  /*! 
    Function to compute logarithm of vector.
    
    \param v vector whose logarithm is to be computed
    \return logarithm of input vector
  */
  HiddenMarkovModels::RealVectorType vlog(const HiddenMarkovModels::RealVectorType& v);

  /*! 
    Function to compute logarithm of matrix.
    
    \param m matrix whose logarithm is to be computed
    \return logarithm of input matrix
  */
  HiddenMarkovModels::RealMatrixType mlog(const HiddenMarkovModels::RealMatrixType& m);

  /*!
    Function to compute the maximum value of a vector and the (first!)  index at which it is attained.
    
    \param u vector whose argmax is sought-for
    \return a tuple of the index of which the max is attained, and the max itself
  */
  boost::tuple<int,
    HiddenMarkovModels::RealType
    > argmax(const HiddenMarkovModels::RealVectorType& u);
};

#endif // HMMUTILS_H
