#ifndef HIDDENMARKOVMODELSNAMESPACE_H
#define HIDDENMARKOVMODELSNAMESPACE_H

/*!
  \file HiddenMarkovModelsNamespace.h
  \brief Header file exposing the whole API.
  \author DOP (dohmatob elvis dopgima)
*/

#include "DiscreteHMM.h" // pull-in HiddenMarkovModels namespace, etc.
#include "HMMBasicTypes.h"  // pull-in basic types (RealType, etc.)
#include "HMMUtils.h" // pull-in load_hmm_matrix, load_hmm_vector, etc.
#include "HMMPathType.h" // pull-in path representation
#include <boost/assign/std/vector.hpp> // for 'operator+='

using namespace boost::assign; // bring 'operator+=()' into scope

#endif // HIDDENMARKOVMODELSNAMESPACE_H
