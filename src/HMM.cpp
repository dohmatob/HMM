/*!
  \file HMM.cpp
  \brief Implementation of HiddenMarkovModels::HMM class
  \author DOP (dohmatob elvis dopgima)
*/

#include "HMM.hpp" // pull-in stuff (namespaces, classes, functions, etc.) to implement
#include "HMMUtils.hpp"

HiddenMarkovModels::HMM::HMM(HiddenMarkovModels::matrix transition,
	 HiddenMarkovModels::matrix emission,
	 HiddenMarkovModels::vector pi)
{
  // sanity checks on matrix dimensions
  BOOST_ASSERT(pi.size() == emission.size1());
  BOOST_ASSERT(transition.size1() == emission.size1());
  BOOST_ASSERT(transition.size2() == emission.size1());

  // set parameters of model
  set_transition(transition);
  set_emission(emission);
  set_pi(pi);
}

void HiddenMarkovModels::HMM::set_transition(const HiddenMarkovModels::matrix& transition)
{
  BOOST_ASSERT(HiddenMarkovModels::is_stochastic_matrix(transition));

  _transition = transition;
  for (int i = 0; i < transition.size1(); i++)
    {
      row(_transition, i) = row(transition, i)/sum(row(transition, i)); // normalization for stochasticity
    }
}

void HiddenMarkovModels::HMM::set_emission(const HiddenMarkovModels::matrix& emission)
{
  BOOST_ASSERT(HiddenMarkovModels::is_stochastic_matrix(emission));

  _emission = emission;
  for (int i = 0; i < emission.size1(); i++)
    {
      row(_emission, i) = row(emission, i)/sum(row(emission, i)); // normalization for stochasticity
    }
}
      
void HiddenMarkovModels::HMM::set_pi(const HiddenMarkovModels::vector& pi)
{
  BOOST_ASSERT(HiddenMarkovModels::is_stochastic_vector(pi));
  
  _pi = pi;
  _pi = pi/sum(pi); // normalization for stochasticity
}

int HiddenMarkovModels::HMM::get_nstates(void) const
{
  return  _pi.size();
}

int HiddenMarkovModels::HMM::get_nsymbols(void) const
{
  return _emission.size2();
}

const HiddenMarkovModels::matrix& HiddenMarkovModels::HMM::get_transition(void) const
{
  return _transition;
}

const HiddenMarkovModels::matrix& HiddenMarkovModels::HMM::get_emission(void) const
{
  return _emission;
}

const HiddenMarkovModels::vector& HiddenMarkovModels::HMM::get_pi(void) const
{
  return _pi;
}

bool HiddenMarkovModels::HMM::is_symbol(unsigned int i) const
{
  return 0 <= i && i < get_nsymbols();
}

boost::tuple<HiddenMarkovModels::sequence_type, // optimal path
	     HiddenMarkovModels::real_type // likelihood of path
	     > HiddenMarkovModels::HMM::viterbi(const HiddenMarkovModels::sequence_type &obseq)
{
  // variables
  int T = obseq.size();
  HiddenMarkovModels::sequence_type hiddenseq(T); // optimal path (sequence of hidden states that generated observed trace)
  HiddenMarkovModels::real_type likelihood;
  HiddenMarkovModels::matrix delta(T, get_nstates());
  ublas::matrix<int> phi(T, get_nstates());

  // logarithms, so we don't suffer underflow!
  HiddenMarkovModels::matrix logtransition = HiddenMarkovModels::mlog(_transition);
  HiddenMarkovModels::matrix logemission = HiddenMarkovModels::mlog(_emission);
  HiddenMarkovModels::vector logpi = HiddenMarkovModels::vlog(_pi);

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
	  HiddenMarkovModels::vector tmp = row(delta, time-1)+column(logtransition, j);
	  boost::tuple<int,
		       HiddenMarkovModels::real_type
		       > x = HiddenMarkovModels::argmax(tmp);
	  BOOST_ASSERT(is_symbol(obseq[time]));
	  delta(time, j) = boost::get<1>(x)+logemission(j, obseq[time]);
	  phi(time, j) = boost::get<0>(x);
	}
    }

  // set last node on optimal path
  HiddenMarkovModels::vector tmp = row(delta, T-1);
  boost::tuple<int,
	       HiddenMarkovModels::real_type
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

  return boost::make_tuple(hiddenseq, likelihood);
}

boost::tuple<HiddenMarkovModels::matrix, // alpha-hat
	     HiddenMarkovModels::matrix, // beta-hat
	     HiddenMarkovModels::matrix, // gamma-hat
	     boost::multi_array<HiddenMarkovModels::real_type,3>, // epsilon-hat
	     HiddenMarkovModels::real_type // likelihood
	     > HiddenMarkovModels::HMM::forward_backward(const HiddenMarkovModels::sequence_type &obseq)
{
  // variables
  unsigned int T = obseq.size();
  HiddenMarkovModels::vector scalers(T); // these things will prevent underflow, etc.
  HiddenMarkovModels::matrix alphatilde(T, get_nstates());
  HiddenMarkovModels::matrix alphahat(T, get_nstates()); // forward variables
  HiddenMarkovModels::matrix betatilde(T, get_nstates()); 
  HiddenMarkovModels::matrix betahat(T, get_nstates()); // backward variables
  HiddenMarkovModels::matrix gammahat(T, get_nstates());
  HiddenMarkovModels::vector tmp(get_nstates());
  boost::multi_array<HiddenMarkovModels::real_type,3> epsilonhat(boost::extents[T-1][get_nstates()][get_nstates()]);
  HiddenMarkovModels::real_type likelihood;

  // compute forward (alpha) variables
#pragma omp parallel for ordered
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
#pragma omp parallel for ordered
  for (int time = T-1; time >= 0; time--)
    {
      if (time == T-1)
	{
	  row(betatilde, time) = ublas::scalar_vector<HiddenMarkovModels::real_type>(get_nstates(), 1);
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
#pragma omp parallel for 
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

boost::tuple<HiddenMarkovModels::HMM, // the learned model
	     HiddenMarkovModels::real_type // the likelihood of the sequences of observations, under this new model
	     > HiddenMarkovModels::HMM::learn(const std::vector<HiddenMarkovModels::sequence_type> &obseqs)
{
  // local typedefs
  typedef boost::multi_array<HiddenMarkovModels::real_type,3> floats3D;
  typedef floats3D::index_range range;
  floats3D::index_gen indices;

  // variables
  HiddenMarkovModels::real_type likelihood;
  HiddenMarkovModels::matrix A(get_nstates(), get_nstates()); // transition matrix for learned model
  HiddenMarkovModels::matrix B(get_nstates(), get_nsymbols()); // emission matrix for learned model
  HiddenMarkovModels::vector pi(get_nstates()); // initial distribution for learned model
  boost::multi_array<HiddenMarkovModels::real_type, 3> u(boost::extents[obseqs.size()][get_nstates()][get_nstates()]);
  boost::multi_array<HiddenMarkovModels::real_type, 3> w(boost::extents[obseqs.size()][get_nstates()][get_nsymbols()]);
  HiddenMarkovModels::matrix v(obseqs.size(), get_nstates());
  HiddenMarkovModels::matrix x(obseqs.size(), get_nstates());
  
  // initializations
  pi = ublas::zero_vector<HiddenMarkovModels::real_type>(get_nstates());
  v = ublas::zero_matrix<HiddenMarkovModels::real_type>(obseqs.size(), get_nstates());
  x = ublas::zero_matrix<HiddenMarkovModels::real_type>(obseqs.size(), get_nstates());
  std::fill(u.data(), u.data()+u.num_elements(), 0);
  std::fill(w.data(), w.data()+w.num_elements(), 0);

  // process all observations (assumed to be independent!)
#pragma omp parallel for
  for (int k = 0; k < obseqs.size(); k++)
    {
      // length of observation
      int T = obseqs[k].size();

      // run Forward-Backward
      boost::tuple<HiddenMarkovModels::matrix, // alpha-hat
		   HiddenMarkovModels::matrix, // beta-hat
		   HiddenMarkovModels::matrix, // gamma-hat
		   boost::multi_array<HiddenMarkovModels::real_type, 3>, // epsilon-hat
		   HiddenMarkovModels::real_type // likelihood
		   >  fb = forward_backward(obseqs[k]);
      HiddenMarkovModels::matrix alphahat = boost::get<0>(fb);
      HiddenMarkovModels::matrix betahat = boost::get<1>(fb);
      HiddenMarkovModels::matrix gammahat = boost::get<2>(fb);
      boost::multi_array<HiddenMarkovModels::real_type, 3> epsilonhat = boost::get<3>(fb);

      // update likelihood
#pragma omp critical
      likelihood += boost::get<4>(fb);

      // calculate auxiliary tensors
      for (int i = 0; i < get_nstates(); i++)
	{
#pragma omp critical
	  pi[i] += gammahat(0, i);
	  for (int time = 0; time < T; time++)
	    {
#pragma omp critical
	      x(k, i) += gammahat(time, i);
	      if (time < T-1)
		{
#pragma omp critical
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
#pragma omp critical
		      w[k][i][j] += gammahat(time, i);
		    }
		}
	    }
	}
    }

  // compute learned model parameters
  pi /= obseqs.size(); // normalization
#pragma omp parallel for
  for (int i = 0; i < get_nstates(); i++)
    {
      HiddenMarkovModels::real_type total1 = sum(column(v, i));
      HiddenMarkovModels::real_type total2 = sum(column(x, i));
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

  return boost::make_tuple(HiddenMarkovModels::HMM(A, B, pi), likelihood);
}

boost::tuple<HiddenMarkovModels::HMM, // the learned model
	     HiddenMarkovModels::real_type // the likelihood of the sequences of observations, under this new model
	     > HiddenMarkovModels::HMM::baum_welch(const std::vector<HiddenMarkovModels::sequence_type> &obseqs,
			       HiddenMarkovModels::real_type tolerance,
			       unsigned int maxiter
			       )
{
  // intializations
  int iteration = 0;
  HiddenMarkovModels::real_type likelihood = -1*std::numeric_limits<HiddenMarkovModels::real_type>::max(); // minus infinity
  HiddenMarkovModels::real_type relative_gain = 0;

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
      boost::tuple<HiddenMarkovModels::HMM,
		   HiddenMarkovModels::real_type
		   > learned = learn(obseqs);
      HMM new_hmm = boost::get<0>(learned);
      real_type new_likelihood = boost::get<1>(learned); 
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

  return boost::make_tuple(HiddenMarkovModels::HMM(_transition, _emission, _pi), likelihood);
}

std::ostream &HiddenMarkovModels::operator<<(std::ostream &cout, HiddenMarkovModels::HMM &hmm)
{
  cout << "transition = " << hmm.get_transition() << "\n\n";
  cout << "emission = " << hmm.get_emission() << "\n\n";
  cout << "pi = " << hmm.get_pi() << "\n\n";

  return cout;
}

HiddenMarkovModels::matrix HiddenMarkovModels::load_hmm_matrix(const char *filename)
{
  // XXX check that filename exists

  std::vector<std::vector<HiddenMarkovModels::real_type> > data;
  std::ifstream input(filename);
  std::string lineData;
  int n = 0;
  int m;

  while(std::getline(input, lineData))
    {
      HiddenMarkovModels::real_type d;
      std::vector<HiddenMarkovModels::real_type> row;
      std::stringstream lineStream(lineData);

      while (lineStream >> d)
	row.push_back(d);

      if (n == 0)
	{
	  m = row.size();
	}

      if (row.size() > 0)
	{
	  if (row.size() != m)
	    {
	      throw "mal-formed matrix line";
	    }

	  data.push_back(row);
	  n++;
	}
    }

  // pack the std array into a ublas matrix
  HiddenMarkovModels::matrix X(n, m);
  for (int i = 0; i < n; i++)
    {
      for (int j = 0; j < m; j++)
	{
	  X(i, j) = data[i][j];

	}
    }

  return X;
}

HiddenMarkovModels::vector HiddenMarkovModels::load_hmm_vector(const char *filename)
{
  // XXX check that filename exists

  return row(HiddenMarkovModels::load_hmm_matrix(filename), 0);
}

std::vector<HiddenMarkovModels::sequence_type> HiddenMarkovModels::load_hmm_observations(const char *filename)
{
  // XXX check that filename exists

  std::vector<HiddenMarkovModels::sequence_type> obseqs;
  std::ifstream input(filename);
  std::string lineData;
  int wordcount = 0;

  while(std::getline(input, lineData))
    {
      int d;
      HiddenMarkovModels::sequence_type row;
      std::stringstream lineStream(lineData);

      while (lineStream >> d)
	row.push_back(d);

      if (row.size() > 0)
	{
	  wordcount++;
	  obseqs.push_back(row);
	}
    }

  return obseqs;
}
