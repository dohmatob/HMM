/*!
  \file HMMUtils.cxx
  \brief Implementation of helper routines defined in HMMUtils.h
  \author DOP (dohmatob elvis dopgima)
*/

#include "HMMUtils.h"

bool HiddenMarkovModels::is_stochastic_vector(const HiddenMarkovModels::RealVectorType& v)
{
  for(int i = 0; i < v.size(); i++)
    {
      if(v[i] < 0)
	{
	  return false;
	}
    }

  return true;
}

bool HiddenMarkovModels::is_stochastic_matrix(const HiddenMarkovModels::RealMatrixType& m)
{
  for(int i = 0; i < m.size1(); i++)
    {
      if(!HiddenMarkovModels::is_stochastic_vector(row(m, i)))
	{
	  return false; // ith row is not stochastic
	}
    }

  return true;
}
      
