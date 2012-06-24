// (c) 2012 DOP (dohmatob elvis dopgima)
// HMM.cpp: principal implementation file

#include "HMM.hpp" // pull-in stuff (namespaces, classes, functions, etc.) to implement

std::ostream &HiddenMarkovModels::operator<<(std::ostream &cout, sequence_type s)
{
  cout << "[" << s.size() << "](";
  for (int i = 0; i < s.size(); i++)
    {
      cout << s[i] << (i < s.size() - 1 ? "," : "");
    }

  cout << ")";

  return cout;
}

ublas::vector<HiddenMarkovModels::real_type> HiddenMarkovModels::vlog(const ublas::vector<HiddenMarkovModels::real_type> &u)
{
  ublas::vector<HiddenMarkovModels::real_type> v(u.size());

  for (int i = 0; i < u.size(); i++)
    {
      v(i) = std::log(u(i));
    }

  return v;
}

ublas::matrix<HiddenMarkovModels::real_type> HiddenMarkovModels::mlog(const ublas::matrix<HiddenMarkovModels::real_type> &A)
{
  ublas::matrix<HiddenMarkovModels::real_type> B(A.size1(), A.size2());

  for (int i = 0; i < A.size1(); i++)
    {
      row(B, i) = HiddenMarkovModels::vlog(row(A, i));
    }

  return B;
}

boost::tuple<int, // index at which max is attained
	     HiddenMarkovModels::real_type // max value
	     > HiddenMarkovModels::argmax(ublas::vector<HiddenMarkovModels::real_type> &u)
{
  int index = 0;
  HiddenMarkovModels::real_type value = u(0);

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

HiddenMarkovModels::HMM::HMM(ublas::matrix<HiddenMarkovModels::real_type> transition,
	 ublas::matrix<HiddenMarkovModels::real_type> emission,
	 ublas::vector<HiddenMarkovModels::real_type> pi)
{
  // sanity checks on matrix dimensions
  BOOST_ASSERT(pi.size()==emission.size1());
  BOOST_ASSERT(transition.size1()==emission.size1());
  BOOST_ASSERT(transition.size2()==emission.size1());

  // sanity checks for stochasticity of parameter matrices
  BOOST_ASSERT(sum(pi) == 1.0);
  for (int i = 0; i < _nhidden; i++)
    {
      BOOST_ASSERT(_pi[i] >= 0);
      BOOST_ASSERT(sum(row(transition, i) == 1.0);
      BOOST_ASSERT(sum(row(emission, i) == 1.0);
      for (int j = 0; j < _nhidden; j++)
	{
	  BOOST_ASSERT(transition(i, j) >= 0);
	}      
      for (int j = 0; j < nsysmbols; j++)
	{
	  BOOST_ASSERT(emission(i, j) >= 0);
	}
    }
  
  _transition = transition;
  _emission = emission;
  _pi = pi;
  _nstates = pi.size();
  _nsymbols = emission.size2();
}

int HiddenMarkovModels::HMM::get_nstates(void)
{
  return _nstates;
}

int HiddenMarkovModels::HMM::get_nsymbols(void)
{
  return _nsymbols;
}

const ublas::matrix<HiddenMarkovModels::real_type>& HiddenMarkovModels::HMM::get_transition(void)
{
  return _transition;
}

const ublas::matrix<HiddenMarkovModels::real_type>& HiddenMarkovModels::HMM::get_emission(void)
{
  return _emission;
}

const ublas::vector<HiddenMarkovModels::real_type>& HiddenMarkovModels::HMM::get_pi(void)
{
  return _pi;
}

bool HiddenMarkovModels::HMM::isSymbol(unsigned int i)
{
  return 0 <= i && i < _nsymbols;
}

boost::tuple<HiddenMarkovModels::sequence_type, // optimal path
	     HiddenMarkovModels::real_type // likelihood of path
	     > HiddenMarkovModels::HMM::viterbi(const HiddenMarkovModels::sequence_type &obseq)
{
  // variables
  int T = obseq.size();
  HiddenMarkovModels::sequence_type hiddenseq(T); // optimal path (sequence of hidden states that generated observed trace)
  HiddenMarkovModels::real_type likelihood;
  ublas::matrix<HiddenMarkovModels::real_type> delta(T, _nstates);
  ublas::matrix<int> phi(T, _nstates);

  // logarithms, so we don't suffer underflow!
  ublas::matrix<HiddenMarkovModels::real_type> logtransition = HiddenMarkovModels::mlog(_transition);
  ublas::matrix<HiddenMarkovModels::real_type> logemission = HiddenMarkovModels::mlog(_emission);
  ublas::vector<HiddenMarkovModels::real_type> logpi = HiddenMarkovModels::vlog(_pi);

  // compute stuff for time = 0
  BOOST_ASSERT(isSymbol(obseq[0]));
  row(delta,0) = logpi + column(logemission, obseq[0]);
  for (int j = 0; j < _nstates; j++)
    {
      phi(0,j) = j;
    }

  // run viterbi proper
  for (int time = 1; time < T; time++)
    {
      for (int j = 0; j < _nstates; j++)
	{
	  ublas::vector<HiddenMarkovModels::real_type> tmp = row(delta, time - 1) + column(logtransition, j);
	  boost::tuple<int,
		       HiddenMarkovModels::real_type
		       > x = HiddenMarkovModels::argmax(tmp);
	  BOOST_ASSERT(isSymbol(obseq[time]));
	  delta(time, j) = boost::get<1>(x) + logemission(j, obseq[time]);
	  phi(time, j) = boost::get<0>(x);
	}
    }

  // set last node on optimal path
  ublas::vector<HiddenMarkovModels::real_type> tmp = row(delta, T - 1);
  boost::tuple<int,
	       HiddenMarkovModels::real_type
	       > x = HiddenMarkovModels::argmax(tmp);
  likelihood = boost::get<1>(x);
  int state = boost::get<0>(x);
  hiddenseq[T-1] = state;

  // backtrack
  for (int time = T-2; time >= 0; time--)
    {
      state = phi(time + 1, state);
      hiddenseq[time] = state;
    }

  return boost::make_tuple(hiddenseq, likelihood);
}

boost::tuple<ublas::matrix<HiddenMarkovModels::real_type>, // alpha-hat
	     ublas::matrix<HiddenMarkovModels::real_type>, // beta-hat
	     ublas::matrix<HiddenMarkovModels::real_type>, // gamma-hat
	     boost::multi_array<HiddenMarkovModels::real_type,3>, // epsilon-hat
	     HiddenMarkovModels::real_type // likelihood
	     > HiddenMarkovModels::HMM::forward_backward(const HiddenMarkovModels::sequence_type &obseq)
{
  // variables
  unsigned int T = obseq.size();
  ublas::vector<HiddenMarkovModels::real_type> scalers(T); // these things will prevent underflow, etc.
  ublas::matrix<HiddenMarkovModels::real_type> alphatilde(T, _nstates);
  ublas::matrix<HiddenMarkovModels::real_type> alphahat(T, _nstates); // forward variables
  ublas::matrix<HiddenMarkovModels::real_type> betatilde(T, _nstates); 
  ublas::matrix<HiddenMarkovModels::real_type> betahat(T, _nstates); // backward variables
  ublas::matrix<HiddenMarkovModels::real_type> gammahat(T, _nstates);
  ublas::vector<HiddenMarkovModels::real_type> tmp(_nstates);
  boost::multi_array<HiddenMarkovModels::real_type,3> epsilonhat(boost::extents[T-1][_nstates][_nstates]);
  HiddenMarkovModels::real_type likelihood;

  // compute forward (alpha) variables
  for (int time = 0; time < T; time++)
    {
      BOOST_ASSERT(isSymbol(obseq[time]));
      if (time == 0)
	{
	  row(alphatilde, time) = element_prod(_pi, column(_emission, obseq[time]));
	}
      else
	{
	  for (int i = 0; i < _nstates; i++)
	    {
	      alphatilde(time, i) = sum(element_prod(row(alphahat, time-1), column(_transition,i))) * _emission(i, obseq[time]);
	    }
	}
      scalers[time] = 1/sum(row(alphatilde, time)); // norm_1(.) equalates to sum(.)
      row(alphahat, time) = scalers[time]*row(alphatilde, time);
    }

  // compute backward (beta) parameters
  for (int time = T-1; time >= 0; time--)
    {
      if (time == T-1)
	{
	  row(betatilde, time) = ublas::scalar_vector<HiddenMarkovModels::real_type>(_nstates, 1);
	}
      else
	{
	  for (int i = 0; i < _nstates; i++)
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
	  for (int i = 0; i < _nstates; i++)
	    {
	      for (int j = 0; j < _nstates; j++)
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
  ublas::matrix<HiddenMarkovModels::real_type> A(_nstates, _nstates); // transition matrix for learned model
  ublas::matrix<HiddenMarkovModels::real_type> B(_nstates, _nsymbols); // emission matrix for learned model
  ublas::vector<HiddenMarkovModels::real_type> pi(_nstates); // initial distribution for learned model
  boost::multi_array<HiddenMarkovModels::real_type, 3> u(boost::extents[obseqs.size()][_nstates][_nstates]);
  boost::multi_array<HiddenMarkovModels::real_type, 3> w(boost::extents[obseqs.size()][_nstates][_nsymbols]);
  ublas::matrix<HiddenMarkovModels::real_type> v(obseqs.size(), _nstates);
  ublas::matrix<HiddenMarkovModels::real_type> x(obseqs.size(), _nstates);
  int k, i, j, time;

  // initializations
  pi = ublas::zero_vector<HiddenMarkovModels::real_type>(_nstates);
  v = ublas::zero_matrix<HiddenMarkovModels::real_type>(obseqs.size(), _nstates);
  x = ublas::zero_matrix<HiddenMarkovModels::real_type>(obseqs.size(), _nstates);
  std::fill(u.data(), u.data()+u.num_elements(), 0);
  std::fill(w.data(), w.data()+w.num_elements(), 0);

  // process all observations (assumed to be independent!)
  for (k = 0; k < obseqs.size(); k++)
    {
      // length of observation
      int T = obseqs[k].size();

      // run Forward-Backward
      boost::tuple<ublas::matrix<HiddenMarkovModels::real_type>, // alpha-hat
		   ublas::matrix<HiddenMarkovModels::real_type>, // beta-hat
		   ublas::matrix<HiddenMarkovModels::real_type>, // gamma-hat
		   boost::multi_array<HiddenMarkovModels::real_type, 3>, // epsilon-hat
		   HiddenMarkovModels::real_type // likelihood
		   >  fb = forward_backward(obseqs[k]);
      ublas::matrix<HiddenMarkovModels::real_type> alphahat = boost::get<0>(fb);
      ublas::matrix<HiddenMarkovModels::real_type> betahat = boost::get<1>(fb);
      ublas::matrix<HiddenMarkovModels::real_type> gammahat = boost::get<2>(fb);
      boost::multi_array<HiddenMarkovModels::real_type, 3> epsilonhat = boost::get<3>(fb);

      // update likelihood
      likelihood += boost::get<4>(fb);

      // calculate auxiliary tensors
      for (i = 0; i < _nstates; i++)
	{
	  pi[i] += gammahat(0, i);
	  for (time = 0; time < T; time++)
	    {
	      x(k, i) += gammahat(time, i);
	      if (time < T-1)
		{
		  v(k, i) += gammahat(time, i);
		  for (j = 0; j < _nstates; j++)
		    {
		      u[k][i][j] += epsilonhat[time][i][j];
		    }
		}
	    }

	  for (j = 0; j < _nsymbols; j++)
	    {
	      for (time = 0; time < T; time++)
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
  for (i = 0; i < _nstates; i++)
    {
      HiddenMarkovModels::real_type total1 = sum(column(v, i));
      HiddenMarkovModels::real_type total2 = sum(column(x, i));
      for (j = 0; j < _nstates; j++)
	{
	  floats3D::array_view<1>::type view1Du = u[indices[range()][i][j]];
	  A(i, j) = std::accumulate(view1Du.begin(), view1Du.end(), 0.0)/total1;
	}
      for (j = 0; j < _nsymbols; j++)
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

      std::cout << "iteration: " << iteration << std::endl;
      iteration++;

      // learn
      boost::tuple<HiddenMarkovModels::HMM,
		   HiddenMarkovModels::real_type
		   > learned = learn(obseqs);
      std::cout << "likelihood = " << boost::get<1>(learned) << std::endl;

      // converged ?
      if (boost::get<1>(learned) == 0.0)
	{
	  std::cout << "CONVERGED." << std::endl;
	  break;
	}

      // update this model
      relative_gain = (boost::get<1>(learned) - likelihood)/abs(boost::get<1>(learned));
      std::cout << "relative gain = " << relative_gain << std::endl;
      _transition = boost::get<0>(learned).get_transition();
      _emission = boost::get<0>(learned).get_emission();
      _pi = boost::get<0>(learned).get_pi();

      // update likehood
      likelihood = boost::get<1>(learned);

      // converged ?
      if (relative_gain < tolerance)
	{

	  std::cout << "CONVERGED." << std::endl;
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

ublas::matrix<HiddenMarkovModels::real_type> HiddenMarkovModels::load_hmm_matrix(const char *filename)
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
  ublas::matrix<HiddenMarkovModels::real_type> X(n, m);
  for (int i = 0; i < n; i++)
    {
      for (int j = 0; j < m; j++)
	{
	  X(i, j) = data[i][j];

	}
    }

  return X;
}

ublas::vector<HiddenMarkovModels::real_type> HiddenMarkovModels::load_hmm_vector(const char *filename)
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
