#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/multi_array.hpp>
#include <boost/assign/std/vector.hpp> // for 'operator+=()'
#include <boost/assert.hpp>
#include <boost/tuple/tuple.hpp> // so I can return multiple values from functions (like in python)
#include <math.h>
#include <iostream>

using namespace boost::numeric::ublas;
using namespace boost::assign;

typedef double real_type;
typedef std::vector<unsigned int> sequence;

std::ostream &operator<<(std::ostream &cout, sequence s)
{
  cout << "[" << s.size() << "](";
  for (int i = 0; i < s.size(); i++)
    {
      cout << s[i] << (i < s.size()-1 ? "," : "");
    }

  cout << ")";

  return cout;
}

typedef struct argmax_struct
{
  real_type value;
  int index;
} argmax_t;

typedef struct path_struct
{
  sequence states;
  real_type likelihood;

  int length(void)
  {
    return states.size();
  }

  int operator[](int i)
  {
    return states[i];
  }
} path_t;

vector<real_type> vlog2(const vector<real_type> &u)
{
  vector<real_type> v(u.size());
  
  for (int i = 0; i < u.size(); i++)
    {
      v(i) = log2(u(i));
    }

  return v;
}

matrix<real_type> mlog2(const matrix<real_type> &A)
{
  matrix<real_type> B(A.size1(),A.size2());
  
  for (int i = 0; i < A.size1(); i++)
    {
      row(B,i) = vlog2(row(A,i));
    }

  return B;
}

argmax_t argmax(vector<real_type> &u)
{
  argmax_t am;
  am.index = 0;
  am.value = u(0);

  for (int i = 1; i < u.size(); i++)
    {
      if (u(i) > am.value)
	{
	  am.value = u(i);
	  am.index = i;
	}
    }

  return am;
}
  
class HMM
{
private:
  int _nhidden;
  int _nobservables;
  matrix<real_type> _transition;
  matrix<real_type> _emission;
  vector<real_type> _pi;

  // logarithms, so we don't suffer 'underflow' in viterbi, etc.
  matrix<real_type> _log2transition;
  matrix<real_type> _log2emission;
  vector<real_type> _log2pi;

public:
  HMM(matrix<real_type> transition, 
      matrix<real_type> emission, 
      vector<real_type> pi);

  const matrix<real_type> &get_transition(void);

  const matrix<real_type> &get_emission(void);

  const vector<real_type> &get_pi(void);

  bool isState(unsigned int i);

  path_t viterbi(const sequence &obseq);

  boost::tuple<matrix<real_type>, // alpha-hat
	       matrix<real_type>, // beta-hat
	       matrix<real_type>, // gamma-hat
	       boost::multi_array<real_type,3>, // epsilon-hat 
	       real_type // likelihood
	       > forward_backward(const sequence &obseq);
  
  boost::tuple<HMM, // the learned model
	       real_type // the likelihood sequence of observations, under this new model
	       > learn(const std::vector<sequence> &obseqs);

  void baum_welch(const std::vector<sequence> &obseqs, 
		 real_type tolerance, 
		 unsigned int maxiter=100
		 );
};

HMM::HMM(matrix<real_type> transition, 
	 matrix<real_type> emission, 
	 vector<real_type> pi)
{
  // sanity checks
  // XXX add checks for stochasticity
  BOOST_ASSERT(pi.size()==emission.size1());
  BOOST_ASSERT(transition.size1()==emission.size1());
  BOOST_ASSERT(transition.size2()==emission.size1());

  _transition = transition;
  _emission = emission;
  _pi = pi;
  _log2transition = mlog2(transition);
  _log2emission = mlog2(emission);
  _log2pi = vlog2(_pi);
  _nhidden = pi.size();
  _nobservables = emission.size2();
}

const matrix<real_type>& HMM::get_transition(void)
{
  return _transition;
}

const matrix<real_type>& HMM::get_emission(void)
{
  return _emission;
}

const vector<real_type>& HMM::get_pi(void)
{
  return _pi;
}

bool HMM::isState(unsigned int i)
{
  return 0 <= i && i < _nhidden;
}

path_t HMM::viterbi(const sequence &obseq)
{
  // initializations
  int T = obseq.size(); 
  sequence hiddenseq(T); // optimal path (sequence of hidden states that generated observed trace)
  path_t path;
  int state;
  real_type proba;
  matrix<real_type> delta(T,_nhidden);
  matrix<int> phi(T,_nhidden);
  argmax_t am;
  vector<real_type> tmp(_nhidden);
  
  // compute stuff for time = 0
  BOOST_ASSERT(isState(obseq[0]));
  row(delta,0) = _log2pi+column(_log2emission,obseq[0]);
  for (int j = 0; j < _nhidden; j++)
    {
      phi(0,j) = j;
    }
  
  // run viterbi proper
  for (int time = 1; time < T; time++)
    {
      for (int j = 0; j < _nhidden; j++)
	{
	  tmp = row(delta,time-1)+column(_log2transition,j);
	  am = argmax(tmp);
	  BOOST_ASSERT(isState(obseq[time]));
	  delta(time,j) = am.value+_log2emission(j,obseq[time]);
	  phi(time,j) = am.index;
	}
    }
  
  // set last node on optimal path
  tmp = row(delta,T-1);
  am = argmax(tmp);
  path.likelihood = am.value;
  state = am.index;
  hiddenseq[T-1] = state;
  
  // backtrack
  for (int time = T-2; time >= 0; time--)
    {
      state = phi(time+1,state);
      hiddenseq[time] = state;
    }
  
  path.states = hiddenseq;
  
  return path;
}

boost::tuple<matrix<real_type>, // alpha-hat
	     matrix<real_type>, // beta-hat
	     matrix<real_type>, // gamma-hat
	     boost::multi_array<real_type,3>, // epsilon-hat 
	     real_type // likelihood
	     > HMM::forward_backward(const sequence &obseq)
{
  // veriables
  unsigned int T = obseq.size();
  vector<real_type> scalers(T); // these things will prevent underflow, etc.
  matrix<real_type> alpha(T,_nhidden);
  matrix<real_type> alphatilde(T,_nhidden);
  matrix<real_type> alphahat(T,_nhidden);
  matrix<real_type> beta(T,_nhidden);
  matrix<real_type> betatilde(T,_nhidden);
  matrix<real_type> betahat(T,_nhidden);
  matrix<real_type> gammahat(T,_nhidden);
  vector<real_type> tmp(_nhidden);
  boost::multi_array<real_type,3> epsilonhat(boost::extents[T-1][_nhidden][_nhidden]);
  real_type likelihood;

  // compute forward (alpha) variables 
  for (int time = 0; time < T; time++)
    {
      BOOST_ASSERT(isState(obseq[time]));
      if (time == 0)
	{
	  row(alphatilde,time) = element_prod(_pi,column(_emission,obseq[time]));
	}
      else
	{
	  for (int i = 0; i < _nhidden; i++)
	    {
	      alphatilde(time,i) = sum(element_prod(row(alphahat,time-1),column(_transition,i)))*_emission(i,obseq[time]);
	    }
	}
      scalers[time] = 1/sum(row(alphatilde,time)); // norm_1(.) equalates to sum(.)
      row(alphahat,time) = scalers[time]*row(alphatilde,time);
    }

  // compute backward (beta) parameters
  for (int time = T-1; time >= 0; time--)
    {
      if (time == T-1)
	{
	  row(betatilde,time) = scalar_vector<real_type>(_nhidden,1);
	}
      else
	{
	  for (int i = 0; i < _nhidden; i++)
	    {
	      betatilde(time,i) = norm_1(element_prod(element_prod(row(_transition,i),column(_emission,obseq[time+1])),row(betahat,time+1)));
	    }
	}
      row(betahat,time) = scalers[time]*row(betatilde,time);
    }

  // compute epsilon and gamma terms
  for (int time = 0; time < T; time++)
    {
      row(gammahat,time) = element_prod(row(alphahat,time),row(betahat,time))/scalers[time];
      if (time < T-1)
	{
	  for (int i = 0; i < _nhidden; i++)
	    {
	      for (int j = 0; j < _nhidden; j++)
		{
		  epsilonhat[time][i][j] = alphahat(time,i)*_transition(i,j)*_emission(j,obseq[time+1])*betahat(time+1,j);
		}
	    }
	}
    }

  likelihood = -1*sum(vlog2(scalers));
  return boost::make_tuple(alphahat,betahat,gammahat,epsilonhat,likelihood);
}

boost::tuple<HMM, // the learned model
	     real_type // the likelihood sequence of observations, under this new model
	     > HMM::learn(const std::vector<sequence> &obseqs)
{
  // local typedefs
  typedef boost::multi_array<real_type,3> floats3D;
  typedef floats3D::index_range range;
  floats3D::index_gen indices;

  // variables 
  real_type likelihood;
  matrix<real_type> A(_nhidden,_nhidden); // transition matrix for learned model
  matrix<real_type> B(_nhidden,_nobservables); // emission matrix for learned model
  vector<real_type> pi(_nhidden); // initial distribution for learned model
  boost::multi_array<real_type,3> u(boost::extents[obseqs.size()][_nhidden][_nhidden]);
  boost::multi_array<real_type,3> w(boost::extents[obseqs.size()][_nhidden][_nobservables]);
  matrix<real_type> v(obseqs.size(),_nhidden);
  matrix<real_type> x(obseqs.size(),_nhidden);
  int k,i,j,time;

  // initializations
  pi = zero_vector<real_type>(_nhidden);
  v = zero_matrix<real_type>(obseqs.size(),_nhidden);
  x = zero_matrix<real_type>(obseqs.size(),_nhidden);
  std::fill(u.data(),u.data()+u.num_elements(),0);
  std::fill(w.data(),w.data()+w.num_elements(),0);
  
  // process all observations
  for (k = 0; k < obseqs.size(); k++)
    {
      // length of observation
      int T = obseqs[k].size();

      // run Forward-Backward 
      boost::tuple<matrix<real_type>, // alpha-hat
		   matrix<real_type>, // beta-hat
		   matrix<real_type>, // gamma-hat
		   boost::multi_array<real_type,3>, // epsilon-hat 
		   real_type // likelihood
		   >  fb = forward_backward(obseqs[k]);
      matrix<real_type> alphahat = boost::get<0>(fb);
      matrix<real_type> betahat = boost::get<1>(fb);
      matrix<real_type> gammahat = boost::get<2>(fb);
      boost::multi_array<real_type,3> epsilonhat = boost::get<3>(fb);

      // update likelihood
      likelihood += boost::get<4>(fb);
      
      // calculate auxiliary tensors
      for (i = 0; i < _nhidden; i++)
	{
	  pi[i] += gammahat(0,i);
	  for (time = 0; time < T; time++)
	    {
	      x(k,i) += gammahat(time,i);
	      if (time < T-1)
		{
		  v(k,i) += gammahat(time,i);		  
		  for (j = 0; j < _nhidden; j++)
		    {
		      u[k][i][j] += epsilonhat[time][i][j];
		    }
		}
	    }

	  for (j = 0; j < _nobservables; j++)
	    {
	      for (time = 0; time < T; time++)
		{
		  if (obseqs[k][time] == j)
		    {
		      w[k][i][j] += gammahat(time,i);
		    }
		}
	    }
	}
    }	 
  
  // compute learned model parameters
  pi /= obseqs.size(); // normalization
  for (i = 0; i < _nhidden; i++)
    {
      real_type total1 = sum(column(v,i));
      real_type total2 = sum(column(x,i));
      for (j = 0; j < _nhidden; j++)
	{
	  floats3D::array_view<1>::type view1Du = u[indices[range()][i][j]];
	  A(i,j) = std::accumulate(view1Du.begin(),view1Du.end(),0.0)/total1;
	}
      for (j = 0; j < _nobservables; j++)
	{
	  floats3D::array_view<1>::type view1Dv = w[indices[range()][i][j]];
	  B(i,j) = std::accumulate(view1Dv.begin(),view1Dv.end(),0.0)/total2;
	}
    }
	  
  return boost::make_tuple(HMM(A,B,pi),likelihood);
}

void HMM::baum_welch(const std::vector<sequence> &obseqs, 
		     real_type tolerance, 
		     unsigned int maxiter
		     )
{
}

std::ostream &operator<<(std::ostream &cout, HMM &hmm)
{
  cout << "transition = " << hmm.get_transition() << "\n\n";
  cout << "emission = " << hmm.get_emission() << "\n\n";
  cout << "pi =" << hmm.get_pi() << "\n\n";

  return cout;
}

std::ostream &operator<<(std::ostream &cout, path_t &path)
{
  cout << "states = " << path.states << std::endl;
  cout << "\nlikelihood = " << path.likelihood;

  return cout;
}

int main(void)
{
  // XXX refactor main into unittest cases
  matrix<real_type> trans(2,2);
  matrix<real_type> em(2,7);
  vector<real_type> pi(2);

  for (int i = 0; i < trans.size1(); i++)
    {
      pi(i) = 1.0/trans.size1();
      for (int j = 0; j < trans.size2(); j++)
	{
	  trans(i,j) = 1.0/trans.size1();
	}
    }

  for (int i = 0; i < em.size1(); i++)
    {
      for (int j = 0; j < em.size2(); j++)
	{
	  em(i,j) = 1.0/em.size2();
	}
    }

  trans(1,0) = 1;
  trans(1,1) = 0;
  vector<real_type> p = prod(pi,trans);
  
  // initialize HMM object
  HMM hmm(trans,em,pi);
  std::cout << "HMM:\n" << hmm << std::endl;

  // run viterbi
  std::vector<sequence> sequences;
  sequence tmp;
  tmp += 0,1,1,1,1,0,1,1,1,1;
  sequences.push_back(tmp);
  tmp.clear();
  tmp += 0,1,1,1,0,1,1,1,1,1;
  sequences.push_back(tmp);
  tmp.clear();
  tmp += 0,1,1,1,1,1,1,1,1,1;
  sequences.push_back(tmp);
  tmp.clear();
  tmp += 0,1,1,1,1,1;
  sequences.push_back(tmp);
  tmp.clear();
  tmp += 0,1,1,1,1,1,1;
  sequences.push_back(tmp);
  tmp.clear();
  tmp += 0,1,1,1,1,1,1,1,1,1;
  sequences.push_back(tmp);
  tmp.clear();
  tmp += 0,1,1,1,1,1,1,1,1,1;
  sequences.push_back(tmp);
  tmp.clear();

  path_t path;
  for (int i = 0; i < sequences.size(); i++)
    {
      path = hmm.viterbi(sequences[i]);
      std::cout << path << std::endl;
    }
  
  boost::multi_array<real_type,3> d(boost::extents[10][2][2]);
  boost::tuple<matrix<real_type>, // alpha-hat
	       matrix<real_type>, // beta-hat
	       matrix<real_type>, // gamma-hat
	       boost::multi_array<real_type,3>, // epsilon-hat 
	       real_type // likelyhood
	       >  fb = hmm.forward_backward(sequences[0]);
  std::cout << boost::get<0>(fb) << std::endl;

  boost::tuple<HMM,
	       real_type
	       > result = hmm.learn(sequences);

  std::cout << boost::get<0>(result) << std::endl;
  return 0;
}
