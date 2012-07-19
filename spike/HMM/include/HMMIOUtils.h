#ifndef HMMIOUTILS_H
#define HMMIOUTILS_H

/*!
  \file HMMIOUtils.h
  \brief Header file for routine functions for i/o business.
  \author DOP (dohmatob elvis dopgima)
*/

#include "DiscreteHMM.h"
#include <iostream> // for std::ostream, etc.

namespace HiddenMarkovModels
{
  /*!
    Overloaded operator<< for HiddenMarkovModels::DiscreteHMM objects.
    \param cout target output stream
    \param dhmm HiddenMarkovModels::DiscreteHMM object to print
  */
  std::ostream& operator<<(std::ostream& cout, const HiddenMarkovModels::DiscreteHMM& dhmm);
};

#endif // HMMIOUTILS_H
