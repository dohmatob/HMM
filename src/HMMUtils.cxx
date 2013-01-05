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

  // XXX check whether sum(v) is 1 

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
       
boost::tuple<int,
	     HiddenMarkovModels::RealType
	     > argmax(const HiddenMarkovModels::RealVectorType& u)
{
  int index = 0;
  HiddenMarkovModels::RealType value = u(0);
  
  for (int i = 0; i < u.size(); i++)
    {
      if (u(i) > value)
	{
	  value = u(i);
	  index = i;
	}
    }

  // render results
  return boost::make_tuple(index,value);
}

HiddenMarkovModels::RealVectorType HiddenMarkovModels::vlog(const HiddenMarkovModels::RealVectorType &u)
{
  // output vector
  HiddenMarkovModels::RealVectorType v(u.size());

  for (int i = 0; i < u.size(); i++)
    {
      v(i) = std::log(u(i));
    }

  return v;
}

HiddenMarkovModels::RealMatrixType HiddenMarkovModels::mlog(const HiddenMarkovModels::RealMatrixType &A)
{
  HiddenMarkovModels::RealMatrixType B(A.size1(), A.size2());

  for (int i = 0; i < A.size1(); i++)
    {
      row(B, i) = HiddenMarkovModels::vlog(row(A, i));
    }

  return B;
}

boost::tuple<int, // index at which max is attained
	     HiddenMarkovModels::RealType // max value
	     > HiddenMarkovModels::argmax(const HiddenMarkovModels::RealVectorType &u)
{
  int index = 0;
  HiddenMarkovModels::RealType value = u(0);

  for (int i = 0; i < u.size(); i++)
    {
      if (u(i) > value)
	{
	  value = u(i);
	  index = i;
	}
    }

  // render results
  return boost::make_tuple(index,value);
}
