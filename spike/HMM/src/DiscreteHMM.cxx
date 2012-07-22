/*!
  \file DiscreteHMM.cxx
  \brief Implementation of DiscreteHMM class.
  \author DOP (dohmatob elvis dopgima)
*/

#include "DiscreteHMM.h"
#include "HMMUtils.h"
#include "HMMPathType.h"
#include <boost/assert.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp> // for constructing matrix proxies like, row, column, etc.

namespace ublas = boost::numeric::ublas;

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

void HiddenMarkovModels::DiscreteHMM::set_transition(const HiddenMarkovModels::RealMatrixType& transition)
{
  BOOST_ASSERT(HiddenMarkovModels::is_stochastic_matrix(transition));

  _transition = transition;
  for (int i = 0; i < get_nstates(); i++)
    {
      row(_transition, i) /= sum(row(_transition, i));
    }
}

void HiddenMarkovModels::DiscreteHMM::set_emission(const HiddenMarkovModels::RealMatrixType& emission)
{
  BOOST_ASSERT(HiddenMarkovModels::is_stochastic_matrix(emission));

  _emission = emission;
  for (int i = 0; i < get_nstates(); i++)
    {
      row(_emission, i) /= sum(row(_emission, i));
    }
}

void HiddenMarkovModels::DiscreteHMM::set_pi(const HiddenMarkovModels::RealVectorType& pi)
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

bool HiddenMarkovModels::DiscreteHMM::is_symbol(unsigned int i) const
{
  return 0 <= i && i < get_nsymbols();
}

HiddenMarkovModels::HMMPathType HiddenMarkovModels::DiscreteHMM::viterbi(const HiddenMarkovModels::ObservationSequenceType& obseq)
{
  // variables
  int T = obseq.size();
  HiddenMarkovModels::ObservationSequenceType hiddenseq(T); // optimal path (sequence of hidden states that generated observed trace)
  HiddenMarkovModels::RealType likelihood;
  HiddenMarkovModels::RealMatrixType delta(T, get_nstates());
  ublas::matrix<int> phi(T, get_nstates());

  // logarithms, so we don't suffer underflow!
  HiddenMarkovModels::RealMatrixType logtransition = HiddenMarkovModels::mlog(_transition);
  HiddenMarkovModels::RealMatrixType logemission = HiddenMarkovModels::mlog(_emission);
  HiddenMarkovModels::RealVectorType logpi = HiddenMarkovModels::vlog(_pi);

  // compute stuff for time = 0
  BOOST_ASSERT(is_symbol(obseq[0]));
  row(delta,0) = logpi + column(logemission, obseq[0]);
  for (int j = 0; j < get_nstates(); j++)
    {
      phi(0,j) = j;
    }

  // run viterbi proper
  for (int time = 1; time < T; time++)
    {
      for (int j = 0; j < get_nstates(); j++)
	{
	  HiddenMarkovModels::RealVectorType tmp = row(delta, time-1)+column(logtransition, j);
	  boost::tuple<int,
		       HiddenMarkovModels::RealType
		       > x = HiddenMarkovModels::argmax(tmp);
	  BOOST_ASSERT(is_symbol(obseq[time]));
	  delta(time, j) = boost::get<1>(x)+logemission(j, obseq[time]);
	  phi(time, j) = boost::get<0>(x);
	}
    }

  // set last node on optimal path
  HiddenMarkovModels::RealVectorType tmp = row(delta, T-1);
  boost::tuple<int,
	       HiddenMarkovModels::RealType
	       > x = HiddenMarkovModels::argmax(tmp);
  likelihood = boost::get<1>(x);
  int state = boost::get<0>(x);
  hiddenseq[T-1] = state;

  // backtrack
  for (int time = T-2; time >= 0; time--)
    {
      state = phi(time+1, state);
      hiddenseq[time] = state;
    }

  return HiddenMarkovModels::HMMPathType(hiddenseq, likelihood);
}

boost::tuple<HiddenMarkovModels::RealMatrixType, // alpha-hat
	     HiddenMarkovModels::RealMatrixType, // beta-hat
	     HiddenMarkovModels::RealMatrixType, // gamma-hat
	     boost::multi_array< HiddenMarkovModels::RealType,3 >, // epsilon-hat
	     HiddenMarkovModels::RealType // likelihood
	     > HiddenMarkovModels::DiscreteHMM::forward_backward(const HiddenMarkovModels::ObservationSequenceType &obseq)
{
  // variables
  unsigned int T = obseq.size();
  HiddenMarkovModels::RealVectorType scalers(T); // these things will prevent underflow, etc.
  HiddenMarkovModels::RealMatrixType alphatilde(T, get_nstates());
  HiddenMarkovModels::RealMatrixType alphahat(T, get_nstates()); // forward variables
  HiddenMarkovModels::RealMatrixType betatilde(T, get_nstates()); 
  HiddenMarkovModels::RealMatrixType betahat(T, get_nstates()); // backward variables
  HiddenMarkovModels::RealMatrixType gammahat(T, get_nstates());
  HiddenMarkovModels::RealVectorType tmp(get_nstates());
  boost::multi_array<HiddenMarkovModels::RealType,3> epsilonhat(boost::extents[T-1][get_nstates()][get_nstates()]);
  HiddenMarkovModels::RealType likelihood;

  // compute forward (alpha) variables
  for (int time = 0; time < T; time++)
    {
      BOOST_ASSERT(is_symbol(obseq[time]));
      if (time == 0)
	{
	  row(alphatilde, time) = element_prod(_pi, column(_emission, obseq[time]));
	}
      else
	{
	  for (int i = 0; i < get_nstates(); i++)
	    {
	      alphatilde(time, i) = sum(element_prod(row(alphahat, time-1), column(_transition,i))) * _emission(i, obseq[time]);
	    }
	}
      scalers[time] = 1/sum(row(alphatilde, time)); 
      row(alphahat, time) = scalers[time]*row(alphatilde, time);
    }

  // compute backward (beta) parameters
  for (int time = T-1; time >= 0; time--)
    {
      if (time == T-1)
	{
	  row(betatilde, time) = ublas::scalar_vector<HiddenMarkovModels::RealType>(get_nstates(), 1);
	}
      else
	{
	  for (int i = 0; i < get_nstates(); i++)
	    {
	      betatilde(time, i) = norm_1(element_prod(element_prod(row(_transition, i), column(_emission, obseq[time+1])), row(betahat, time+1)));
	    }
	}
      row(betahat, time) = scalers[time]*row(betatilde, time);
    }

  // compute epsilon and gamma terms
  for (int time = 0; time < T; time++)
    {
      row(gammahat, time) = element_prod(row(alphahat, time), row(betahat, time))/scalers[time];
      if (time < T-1)
	{
	  for (int i = 0; i < get_nstates(); i++)
	    {
	      for (int j = 0; j < get_nstates(); j++)
		{
		  epsilonhat[time][i][j] = alphahat(time, i)*_transition(i, j)*_emission(j, obseq[time+1])*betahat(time+1, j);
		}
	    }
	}
    }

  // compute likelihood
  likelihood = -1*sum(HiddenMarkovModels::vlog(scalers));

  return boost::make_tuple(alphahat, betahat,gammahat, epsilonhat, likelihood);
}

boost::tuple<HiddenMarkovModels::DiscreteHMM, // the learned model
	     HiddenMarkovModels::RealType // the likelihood of the sequences of observations, under this new model
	     > HiddenMarkovModels::DiscreteHMM::learn(const std::vector< HiddenMarkovModels::ObservationSequenceType > &obseqs)
{
  // local typedefs
  typedef boost::multi_array<HiddenMarkovModels::RealType,3> floats3D;
  typedef floats3D::index_range range;
  floats3D::index_gen indices;

  // variables
  HiddenMarkovModels::RealType likelihood;
  HiddenMarkovModels::RealMatrixType A(get_nstates(), get_nstates()); // transition matrix for learned model
  HiddenMarkovModels::RealMatrixType B(get_nstates(), get_nsymbols()); // emission matrix for learned model
  HiddenMarkovModels::RealVectorType pi(get_nstates()); // initial distribution for learned model
  boost::multi_array< HiddenMarkovModels::RealType, 3 > u(boost::extents[obseqs.size()][get_nstates()][get_nstates()]);
  boost::multi_array< HiddenMarkovModels::RealType, 3 > w(boost::extents[obseqs.size()][get_nstates()][get_nsymbols()]);
  HiddenMarkovModels::RealMatrixType v(obseqs.size(), get_nstates());
  HiddenMarkovModels::RealMatrixType x(obseqs.size(), get_nstates());
  
  // initializations
  pi = ublas::zero_vector< HiddenMarkovModels::RealType >(get_nstates());
  v = ublas::zero_matrix< HiddenMarkovModels::RealType >(obseqs.size(), get_nstates());
  x = ublas::zero_matrix< HiddenMarkovModels::RealType >(obseqs.size(), get_nstates());
  std::fill(u.data(), u.data()+u.num_elements(), 0);
  std::fill(w.data(), w.data()+w.num_elements(), 0);

  // process all observations (assumed to be independent!)
  for (int k = 0; k < obseqs.size(); k++)
    {
      // length of observation
      int T = obseqs[k].size();

      // run Forward-Backward
      boost::tuple<HiddenMarkovModels::RealMatrixType, // alpha-hat
		   HiddenMarkovModels::RealMatrixType, // beta-hat
		   HiddenMarkovModels::RealMatrixType, // gamma-hat
		   boost::multi_array<HiddenMarkovModels::RealType, 3>, // epsilon-hat
		   HiddenMarkovModels::RealType // likelihood
		   >  fb = forward_backward(obseqs[k]);
      HiddenMarkovModels::RealMatrixType alphahat = boost::get<0>(fb);
      HiddenMarkovModels::RealMatrixType betahat = boost::get<1>(fb);
      HiddenMarkovModels::RealMatrixType gammahat = boost::get<2>(fb);
      boost::multi_array<HiddenMarkovModels::RealType, 3> epsilonhat = boost::get<3>(fb);

      // update likelihood
      likelihood += boost::get<4>(fb);

      // calculate auxiliary tensors
      for (int i = 0; i < get_nstates(); i++)
	{
	  pi[i] += gammahat(0, i);
	  for (int time = 0; time < T; time++)
	    {
	      x(k, i) += gammahat(time, i);
	      if (time < T-1)
		{
		  v(k, i) += gammahat(time, i);
		  for (int j = 0; j < get_nstates(); j++)
		    {
		      u[k][i][j] += epsilonhat[time][i][j];
		    }
		}
	    }

	  for (int j = 0; j < get_nsymbols(); j++)
	    {
	      for (int time = 0; time < T; time++)
		{
		  if (obseqs[k][time] == j)
		    {
		      w[k][i][j] += gammahat(time, i);
		    }
		}
	    }
	}
    }

  // compute learned model parameters
  pi /= obseqs.size(); // normalization
  for (int i = 0; i < get_nstates(); i++)
    {
      HiddenMarkovModels::RealType total1 = sum(column(v, i));
      HiddenMarkovModels::RealType total2 = sum(column(x, i));
      for (int j = 0; j < get_nstates(); j++)
	{
	  floats3D::array_view<1>::type view1Du = u[indices[range()][i][j]];
	  A(i, j) = std::accumulate(view1Du.begin(), view1Du.end(), 0.0)/total1;
	}
      for (int j = 0; j < get_nsymbols(); j++)
	{
	  floats3D::array_view<1>::type view1Dv = w[indices[range()][i][j]];
	  B(i, j) = std::accumulate(view1Dv.begin(), view1Dv.end(), 0.0)/total2;
	}
    }

  return boost::make_tuple(HiddenMarkovModels::DiscreteHMM(A, B, pi), likelihood);
}

HiddenMarkovModels::RealType HiddenMarkovModels::DiscreteHMM::baum_welch(const std::vector< HiddenMarkovModels::ObservationSequenceType > &obseqs,
									 HiddenMarkovModels::RealType tolerance,
									 unsigned int maxiter
									 )
{
  // intializations
  int iteration = 0;
  HiddenMarkovModels::RealType likelihood = -1*std::numeric_limits< HiddenMarkovModels::RealType >::max(); // minus infinity
  HiddenMarkovModels::RealType relative_gain = 0;

  // main loop
  while (true)
    {
      // done ?
      if (iteration > maxiter)
	{
	  std::cout << "OUT OF COMPUTATION BUDGET" << std::endl;
	  break;
	}

      std::cout << std::endl;
      std::cout << "Iteration: " << iteration << std::endl;
      iteration++;

      // learn
      boost::tuple<HiddenMarkovModels::DiscreteHMM,
		   HiddenMarkovModels::RealType
		   > learned_hmm = learn(obseqs);
      DiscreteHMM new_hmm = boost::get<0>(learned_hmm);
      RealType new_likelihood = boost::get<1>(learned_hmm); 
      std::cout << "\tLikelihood: " << new_likelihood << std::endl;

      // converged ?
      if (new_likelihood == 0.0)
	{
	  std::cout << "\tCONVERGED." << std::endl;
	  break;
	}

      // update this model
      relative_gain = (new_likelihood - likelihood)/abs(new_likelihood);
      BOOST_ASSERT(relative_gain >= 0); // if this fails, then something is terribly wrong with the implementation!
      std::cout << "\tRelative gain in likelihood: " << relative_gain << std::endl;
      set_transition(new_hmm.get_transition());
      set_emission(new_hmm.get_emission());
      set_pi(new_hmm.get_pi());

      // update likehood
      likelihood = new_likelihood;

      // converged ?
      if (relative_gain < tolerance)
	{
	  std::cout << "\tCONVERGED (tolerance was set to " << tolerance << ")." << std::endl;
	  break;
	}
    }

  return likelihood;
}
