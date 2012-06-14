// (c) 2012 DOP (dohmatob elvis dopgima)

#include <boost/numeric/ublas/matrix.hpp> 
#include <boost/numeric/ublas/matrix_proxy.hpp> // matrix row, matrix colum, etc.
#include <boost/numeric/ublas/io.hpp> // display matrix, etc.
#include <boost/multi_array.hpp> // for multi-dimensional arrays (aka tensors), etc.
#include <boost/assign/std/vector.hpp> // operator+=() for vectors, etc.
#include <boost/assert.hpp> 
#include <boost/tuple/tuple.hpp> // so I can return multiple values from functions (like in python)
#include <math.h> // log, abs, etc.
#include <iostream> // cout, etc.
#include <fstream> // file handling functions like getline, etc.
#include <sstream>
#include <algorithm> // random_shuffle, etc.
#include <ctype.h> //__toascii
#include <stdio.h> // printf

using namespace boost::numeric::ublas;
using namespace boost::assign;

typedef double real_type;
typedef std::vector<unsigned int> sequence_type;

std::ostream &operator<<(std::ostream &cout, sequence_type s)
{
  cout << "[" << s.size() << "](";
  for (int i = 0; i < s.size(); i++)
    {
      cout << s[i] << (i < s.size()-1 ? "," : "");
    }

  cout << ")";

  return cout;
}

vector<real_type> vlog(const vector<real_type> &u)
{
  vector<real_type> v(u.size());
  
  for (int i = 0; i < u.size(); i++)
    {
      v(i) = std::log(u(i));
    }

  return v;
}

matrix<real_type> mlog(const matrix<real_type> &A)
{
  matrix<real_type> B(A.size1(),A.size2());
  
  for (int i = 0; i < A.size1(); i++)
    {
      row(B,i) = vlog(row(A,i));
    }

  return B;
}

boost::tuple<int,
	     real_type
	     > argmax(vector<real_type> &u)
{
  int index = 0;
  real_type value = u(0);

  for (int i = 0; i < u.size(); i++)
    {
      if (u(i) > value)
	{
	  value = u(i);
	  index = i;
	}
    }

  return boost::make_tuple(index,value);
}
  
class HMM
{
private:
  int _nstates;
  int _nsymbols;
  matrix<real_type> _transition;
  matrix<real_type> _emission;
  vector<real_type> _pi;

public:
  HMM(matrix<real_type> transition, 
      matrix<real_type> emission, 
      vector<real_type> pi);

  int get_nstates(void);

  int get_nsymbols(void);

  const matrix<real_type> &get_transition(void);

  const matrix<real_type> &get_emission(void);

  const vector<real_type> &get_pi(void);

  bool isSymbol(unsigned int i);

  boost::tuple<sequence_type, // optimal path
	       real_type // likelihood of path
	       > viterbi(const sequence_type &obseq);

  boost::tuple<matrix<real_type>, // alpha-hat
	       matrix<real_type>, // beta-hat
	       matrix<real_type>, // gamma-hat
	       boost::multi_array<real_type,3>, // epsilon-hat 
	       real_type // likelihood
	       > forward_backward(const sequence_type &obseq);
  
  boost::tuple<HMM, // the learned model
	       real_type // the likelihood of sequences of observations, under this new model
	       > learn(const std::vector<sequence_type> &obseqs);

  boost::tuple<HMM, // the learned model
	       real_type // the likelihood of sequences of observations, under this new model
	       > baum_welch(const std::vector<sequence_type> &obseqs, //
			    real_type tolerance=1e-9, // tolerance level for convergence 
			    unsigned int maxiter=200 // maximum number of iterations
			    );
};

HMM::HMM(matrix<real_type> transition, 
	 matrix<real_type> emission, 
	 vector<real_type> pi)
{
  // sanity checks
  BOOST_ASSERT(pi.size()==emission.size1());
  BOOST_ASSERT(transition.size1()==emission.size1());
  BOOST_ASSERT(transition.size2()==emission.size1());

  // XXX add checks for stochasticity

  _transition = transition;
  _emission = emission;
  _pi = pi;
  _nstates = pi.size();
  _nsymbols = emission.size2();
}

int HMM::get_nstates(void)
{
  return _nstates;
}

int HMM::get_nsymbols(void)
{
  return _nsymbols;
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

bool HMM::isSymbol(unsigned int i)
{
  return 0 <= i && i < _nsymbols;
}

boost::tuple<sequence_type, // optimal path
	     real_type // likelihood of path
	     > HMM::viterbi(const sequence_type &obseq)
{
  // variables
  int T = obseq.size(); 
  sequence_type hiddenseq(T); // optimal path (sequence of hidden states that generated observed trace)
  real_type likelihood;
  matrix<real_type> delta(T,_nstates);
  matrix<int> phi(T,_nstates);
  vector<real_type> tmp(_nstates);
  
  // logarithms, so we don't suffer underflow!
  matrix<real_type> logtransition = mlog(_transition);
  matrix<real_type> logemission = mlog(_emission);
  vector<real_type> logpi = vlog(_pi);

  // compute stuff for time = 0
  BOOST_ASSERT(isSymbol(obseq[0]));
  row(delta,0) = logpi+column(logemission,obseq[0]);
  for (int j = 0; j < _nstates; j++)
    {
      phi(0,j) = j;
    }
  
  // run viterbi proper
  for (int time = 1; time < T; time++)
    {
      for (int j = 0; j < _nstates; j++)
	{
	  tmp = row(delta,time-1)+column(logtransition,j);
	  boost::tuple<int,
		       real_type
		       > x = argmax(tmp);
	  BOOST_ASSERT(isSymbol(obseq[time]));
	  delta(time,j) = boost::get<1>(x)+logemission(j,obseq[time]);
	  phi(time,j) = boost::get<0>(x);
	}
    }
  
  // set last node on optimal path
  tmp = row(delta,T-1);
  boost::tuple<int,
	       real_type
	       > x = argmax(tmp);
  likelihood = boost::get<1>(x);
  int state = boost::get<0>(x);
  hiddenseq[T-1] = state;
  
  // backtrack
  for (int time = T-2; time >= 0; time--)
    {
      state = phi(time+1,state);
      hiddenseq[time] = state;
    }
  
  return boost::make_tuple(hiddenseq,likelihood);
}

boost::tuple<matrix<real_type>, // alpha-hat
	     matrix<real_type>, // beta-hat
	     matrix<real_type>, // gamma-hat
	     boost::multi_array<real_type,3>, // epsilon-hat 
	     real_type // likelihood
	     > HMM::forward_backward(const sequence_type &obseq)
{
  // veriables
  unsigned int T = obseq.size();
  vector<real_type> scalers(T); // these things will prevent underflow, etc.
  matrix<real_type> alpha(T,_nstates);
  matrix<real_type> alphatilde(T,_nstates);
  matrix<real_type> alphahat(T,_nstates);
  matrix<real_type> beta(T,_nstates);
  matrix<real_type> betatilde(T,_nstates);
  matrix<real_type> betahat(T,_nstates);
  matrix<real_type> gammahat(T,_nstates);
  vector<real_type> tmp(_nstates);
  boost::multi_array<real_type,3> epsilonhat(boost::extents[T-1][_nstates][_nstates]);
  real_type likelihood;

  // compute forward (alpha) variables 
  for (int time = 0; time < T; time++)
    {
      BOOST_ASSERT(isSymbol(obseq[time]));
      if (time == 0)
	{
	  row(alphatilde,time) = element_prod(_pi,column(_emission,obseq[time]));
	}
      else
	{
	  for (int i = 0; i < _nstates; i++)
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
	  row(betatilde,time) = scalar_vector<real_type>(_nstates,1);
	}
      else
	{
	  for (int i = 0; i < _nstates; i++)
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
	  for (int i = 0; i < _nstates; i++)
	    {
	      for (int j = 0; j < _nstates; j++)
		{
		  epsilonhat[time][i][j] = alphahat(time,i)*_transition(i,j)*_emission(j,obseq[time+1])*betahat(time+1,j);
		}
	    }
	}
    }

  // compute likelihood
  likelihood = -1*sum(vlog(scalers));

  return boost::make_tuple(alphahat,betahat,gammahat,epsilonhat,likelihood);
}

boost::tuple<HMM, // the learned model
	     real_type // the likelihood sequence of observations, under this new model
	     > HMM::learn(const std::vector<sequence_type> &obseqs)
{
  // local typedefs
  typedef boost::multi_array<real_type,3> floats3D;
  typedef floats3D::index_range range;
  floats3D::index_gen indices;

  // variables 
  real_type likelihood;
  matrix<real_type> A(_nstates,_nstates); // transition matrix for learned model
  matrix<real_type> B(_nstates,_nsymbols); // emission matrix for learned model
  vector<real_type> pi(_nstates); // initial distribution for learned model
  boost::multi_array<real_type,3> u(boost::extents[obseqs.size()][_nstates][_nstates]);
  boost::multi_array<real_type,3> w(boost::extents[obseqs.size()][_nstates][_nsymbols]);
  matrix<real_type> v(obseqs.size(),_nstates);
  matrix<real_type> x(obseqs.size(),_nstates);
  int k,i,j,time;

  // initializations
  pi = zero_vector<real_type>(_nstates);
  v = zero_matrix<real_type>(obseqs.size(),_nstates);
  x = zero_matrix<real_type>(obseqs.size(),_nstates);
  std::fill(u.data(),u.data()+u.num_elements(),0);
  std::fill(w.data(),w.data()+w.num_elements(),0);
  
  // process all observations (assumed to be independent!)
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
      for (i = 0; i < _nstates; i++)
	{
	  pi[i] += gammahat(0,i);
	  for (time = 0; time < T; time++)
	    {
	      x(k,i) += gammahat(time,i);
	      if (time < T-1)
		{
		  v(k,i) += gammahat(time,i);		  
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
		      w[k][i][j] += gammahat(time,i);
		    }
		}
	    }
	}
    }	 
  
  // compute learned model parameters
  pi /= obseqs.size(); // normalization
  for (i = 0; i < _nstates; i++)
    {
      real_type total1 = sum(column(v,i));
      real_type total2 = sum(column(x,i));
      for (j = 0; j < _nstates; j++)
	{
	  floats3D::array_view<1>::type view1Du = u[indices[range()][i][j]];
	  A(i,j) = std::accumulate(view1Du.begin(),view1Du.end(),0.0)/total1;
	}
      for (j = 0; j < _nsymbols; j++)
	{
	  floats3D::array_view<1>::type view1Dv = w[indices[range()][i][j]];
	  B(i,j) = std::accumulate(view1Dv.begin(),view1Dv.end(),0.0)/total2;
	}
    }
	  
  return boost::make_tuple(HMM(A,B,pi),likelihood);
}

boost::tuple<HMM, // the learned model
	     real_type // the likelihood sequence of observations, under this new model
	     > HMM::baum_welch(const std::vector<sequence_type> &obseqs, 
			       real_type tolerance, 
			       unsigned int maxiter
			       )
{
  // intializations
  int iteration = 0;
  real_type likelihood = -1*std::numeric_limits<real_type>::max(); // minus infinity
  real_type relative_gain = 0;

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
      boost::tuple<HMM, 
		   real_type
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

  return boost::make_tuple(HMM(_transition,_emission,_pi),likelihood);
}

std::ostream &operator<<(std::ostream &cout, HMM &hmm)
{
  cout << "transition = " << hmm.get_transition() << "\n\n";
  cout << "emission = " << hmm.get_emission() << "\n\n";
  cout << "pi = " << hmm.get_pi() << "\n\n";

  return cout;
}

matrix<real_type> load_hmm_matrix(const char *filename)
{
  // XXX check that filename exists

  std::vector<std::vector<real_type> > data;
  std::ifstream input(filename);
  std::string lineData;
  int n = 0;
  int m;

  while(std::getline(input, lineData))
    {
      real_type d;
      std::vector<real_type> row;
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
  
  matrix<real_type> X(n,m);
  for (int i = 0; i < n; i++)
    {
      for (int j = 0; j < m; j++)
	{
	  X(i,j) = data[i][j];

	}
    }

  return X;
}

vector<real_type> load_hmm_vector(const char *filename)
{
  // XXX check that filename exists

  return row(load_hmm_matrix(filename),0);
}

std::vector<sequence_type> load_hmm_observations(const char *filename)
{
  // XXX check that filename exists

  std::vector<sequence_type> sequences;
  std::ifstream input(filename);
  std::string lineData;
  int wordcount = 0;

  while(std::getline(input, lineData))
    {
      int d;
      sequence_type row;
      std::stringstream lineStream(lineData);
      
      while (lineStream >> d)
	row.push_back(d);
      
      if (row.size() > 0)
	{
	  wordcount++;
	  sequences.push_back(row);
	}
    }
 
  return sequences;
}

int main(void)
{
  // XXX refactor main into unittest cases
  std::cout << "Loadin: HMM parameters from files .." << std::endl;
  matrix<real_type> trans = load_hmm_matrix("corpus_transition.dat");
  matrix<real_type> em = load_hmm_matrix("corpus_emission.dat");
  vector<real_type> pi = load_hmm_vector("corpus_pi.dat");
  std::cout << "Done.\n" << std::endl;  
  
  // initialize HMM object
  HMM hmm(trans,em,pi);
  std::cout << "HMM:\n" << hmm;

  // prepare data
  std::cout << "Loadin: english words from corpus file .." << std::endl;
  std::vector<sequence_type> corpus = load_hmm_observations("corpus_words.dat"); // load
  std::cout << "Done (loaded " << corpus.size() << " words).\n" << std::endl;


  // draw a random sample
  int nlessons = 1000;
  std::cout << "Sampling " << nlessons << " words from corpus .." << std::endl;
  std::random_shuffle(corpus.begin(), corpus.end()); 
  std::vector<sequence_type> lessons = std::vector<sequence_type>(corpus.begin(),corpus.begin()+nlessons%corpus.size()); 
  std::cout << "Done.\n" << std::endl;

  boost::tuple<sequence_type,
	       real_type
	       > path = hmm.viterbi(lessons[2]);

  std::cout << "The a posteriori most probable sequence of hidden states that generated the trace " << lessons[2] << " is " << boost::get<0>(path) << "." << std::endl;
  std::cout << "Its (log) likelihood is " << boost::get<1>(path) << ".\n" << std::endl;
    
  // Bauw-Welch
  hmm.baum_welch(lessons);  
  std::cout << "\nFinal HMM:\n" << hmm;

  std::cout << "Viterbi classification of the 26 symbols (cf. letters of the english alphabet):" << std::endl;
  sequence_type symbol(1);
  unsigned int correction;
  for (int j = 0; j < 26; j++)
    {
      symbol[0] = j;
      boost::tuple<sequence_type,
		   real_type
		   > path = hmm.viterbi(symbol);

      unsigned int which = boost::get<0>(path)[0]; // vowel or consonant ?
      
      // let's call a's cluster "vowel" and call the other cluster "consonant"
      if (j == 0)
	{
	  correction = which;
	}
      which = correction ? 1 - which : which;
      
      printf("\t%c is a %s\n", __toascii('A')+j,which?"consonant":"vowel");
      // std::cout << "\t" << j << " is in class " << boost::get<0>(path)[0] << std::endl;
    }
      
  return 0;
}
