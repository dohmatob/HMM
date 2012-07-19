#ifndef HMMUTILS_H
#define HMMUTILS_H

/*!
  \file HMMUtils.h
  \brief Header file for routine function..
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
};

#endif // HMMUTILS_H
