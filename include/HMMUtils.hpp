#ifndef HMMUTILS_H
#define HMMUTILS_H

/*!
  \file HMMUtils.hpp
  \brief Header file for routine functions for manipulating (loading matrices from files, etc.) HMM objects
  \author DOP (dohmatob elvis dopgima)
*/

#include "HMMBasicTypes.hpp"

namespace HiddenMarkovModels
{
  /*!
    Overloading of operator<< for HiddenMarkovModels::sequence_type
    
    \param cout - output stream to receive flux
    @param seq - sequence to be displayed
  */
  std::ostream &operator<<(std::ostream &cout, sequence_type seq);

  /*!
    Function to compute logarithm of vector.
    
    @param u - vector whose logarithm is to be computed
    @return Logarithm of input vector
  */
  vector vlog(const vector &u);
  
  /*! 
    Function to compute logarithm of matrix.
    
    @param m - matrix whose logarithm is to be computed
    @return logarithm of input matrix
  */
  matrix mlog(const matrix &m);

  /*!
    Function to check whether matrix is stochastic.
    
    @param m - matrix to be checked for stochasticity
    @return true if matrix if stochastic, false otherwise
  */
  bool is_stochastic_matrix(const matrix& m);

  /**
     Function to check whether vector is stochastic.
     
     @param v - vector to be checked for stochasticity
     @return true if vector if stochastic, false otherwise
  **/
  bool is_stochastic_vector(const vector& v);

  /** @brief Class to incapsulate Hidden Markov Models
      
      @author DOP (dohmatob elvis dopgima)
  **/

  /**
     Function to compute the maximum value of a vector and the (first!)  index at which it is attained.

     @param u - vector whose argmax is sought-for
     @return - a tuple of the index of which the max is attained, and the max itself
   **/
  boost::tuple<int,
	       real_type
	       > argmax(vector &u);
}

#endif // HMMUTILS_H
