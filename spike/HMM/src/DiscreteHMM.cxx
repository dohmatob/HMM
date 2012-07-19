/*!
  \file DiscreteHMM.cxx
  \brief Implementation of DiscreteHMM class.
  \author DOP (dohmatob elvis dopgima)
*/

#include "DiscreteHMM.h"
#include "HMMUtils.h"
#include <boost/assert.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp> // for constructing matrix proxies like, row, column, etc.

HiddenMarkovModels::DiscreteHMM::DiscreteHMM(HiddenMarkovModels::RealMatrixType& transition, 
					     HiddenMarkovModels::RealMatrixType& emission, 
					     HiddenMarkovModels::RealVectorType& pi)
{
  // sanity checks on dimensions of matrices
  BOOST_ASSERT(transition.size1() == transition.size2()); // transition matrix should be square
  BOOST_ASSERT(pi.size() == transition.size1()); // transition matrix should have as many rows as there are hidden states in the model
  BOOST_ASSERT(transition.size1() == emission.size1()); // transition and emission matrices should have same number of rows
  // initialize model parameters proper
  set_transition(transition);
  set_emission(emission);
  set_pi(pi);
}

void HiddenMarkovModels::DiscreteHMM::set_transition(HiddenMarkovModels::RealMatrixType& transition)
{
  BOOST_ASSERT(HiddenMarkovModels::is_stochastic_matrix(transition));

  _transition = transition;
  for (int i = 0; i < get_nstates(); i++)
    {
      row(_transition, i) /= sum(row(_transition, i));
    }
}

void HiddenMarkovModels::DiscreteHMM::set_emission(HiddenMarkovModels::RealMatrixType& emission)
{
  BOOST_ASSERT(HiddenMarkovModels::is_stochastic_matrix(emission));

  _emission = emission;
  for (int i = 0; i < get_nstates(); i++)
    {
      row(_emission, i) /= sum(row(_emission, i));
    }
}

void HiddenMarkovModels::DiscreteHMM::set_pi(HiddenMarkovModels::RealVectorType& pi)
{
  BOOST_ASSERT(HiddenMarkovModels::is_stochastic_vector(pi));

  _pi = pi;
  _pi /= sum(_pi);
}

int HiddenMarkovModels::DiscreteHMM::get_nstates() const
{
  return _pi.size();
}

int HiddenMarkovModels::DiscreteHMM::get_nsymbols() const
{
  return _emission.size2();
}

const HiddenMarkovModels::RealMatrixType& HiddenMarkovModels::DiscreteHMM::get_transition() const
{
  return _transition;
}

const HiddenMarkovModels::RealMatrixType& HiddenMarkovModels::DiscreteHMM::get_emission() const
{
  return _emission;
}

const HiddenMarkovModels::RealVectorType& HiddenMarkovModels::DiscreteHMM::get_pi() const
{
  return _pi;
}

