/*!
  \file HMMUtils.cpp
  \brief Implementation of HMMUtils.hpp
  \author DOP (dohmatob elvis dopgima)
*/

#include "HMMUtils.hpp"

std::ostream &HiddenMarkovModels::operator<<(std::ostream &cout, SequenceType s)
{
  cout << "[" << s.size() << "](";
  for (int i = 0; i < s.size(); i++)
    {
      cout << s[i] << (i < s.size()-1 ? "," : "");
    }

  cout << ")";

  return cout;
}

/**
 * method to compute logarithm of vector
 **/
HiddenMarkovModels::vector HiddenMarkovModels::vlog(const HiddenMarkovModels::vector &u)
{
  // output vector
  HiddenMarkovModels::vector v(u.size());

  for (int i = 0; i < u.size(); i++)
    {
      v(i) = std::log(u(i));
    }

  return v;
}

HiddenMarkovModels::matrix HiddenMarkovModels::mlog(const HiddenMarkovModels::matrix &A)
{
  HiddenMarkovModels::matrix B(A.size1(), A.size2());

  for (int i = 0; i < A.size1(); i++)
    {
      row(B, i) = HiddenMarkovModels::vlog(row(A, i));
    }

  return B;
}

boost::tuple<int, // index at which max is attained
	     HiddenMarkovModels::RealType // max value
	     > HiddenMarkovModels::argmax(HiddenMarkovModels::vector &u)
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

bool HiddenMarkovModels::is_stochastic_vector(const HiddenMarkovModels::vector& v)
{
  for (int i = 0; i < v.size(); i++)
    {
      if(v[i] < 0)
	{
	  return false;
	}
    }

  return true;
}

bool HiddenMarkovModels::is_stochastic_matrix(const HiddenMarkovModels::matrix& m)
{
  for (int i = 0; i < m.size1(); i++)
    {
      if(!is_stochastic_vector(row(m, i)))
	{
	  return false;
	}
    }

  return true;
}

