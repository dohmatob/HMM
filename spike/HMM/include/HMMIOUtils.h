#ifndef HMMIOUTILS_H
#define HMMIOUTILS_H

/*!
  \file HMMIOUtils.h
  \brief Header file for routine functions for i/o business.
  \author DOP (dohmatob elvis dopgima)
*/

#include "DiscreteHMM.h"
#include <fstream>
#include <iostream> // for std::ostream, etc.

namespace HiddenMarkovModels
{
  /*!
    Overloaded operator<< for HiddenMarkovModels::DiscreteHMM objects.
    \param cout target output stream
    \param dhmm HiddenMarkovModels::DiscreteHMM object to print
  */
  std::ostream& operator<<(std::ostream& cout, const HiddenMarkovModels::DiscreteHMM& dhmm);

  /*!
    Overloading of operator<< for HiddenMarkovModels::SequenceType
    
    \param cout output stream to receive flux
    \param seq sequence to be displayed
  */
  std::ostream &operator<<(std::ostream &cout, const HiddenMarkovModels::ObservationSequenceType seq);

  /*!
    Function to load a matrix from a file.
    
    \param filename name of file containing matrix to be read
    \return read matrix
  */
  HiddenMarkovModels::RealMatrixType load_hmm_matrix(const char *filename);
  
  /*! 
    Function to load a vector from a file.
   */
  HiddenMarkovModels::RealVectorType load_hmm_vector(const char *filename);
  
  /*! 
    Function to load a sequence of observations from a file.
  */
  std::vector< HiddenMarkovModels::ObservationSequenceType > load_hmm_observations(const char *filename);
};

#endif // HMMIOUTILS_H
