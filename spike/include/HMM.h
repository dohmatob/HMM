// (c) 2012 DOP (dohmatob elvis dopgima)
// HMM.h: principal header file

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

using namespace boost::numeric::ublas;
using namespace boost::assign;

typedef double real_type;
typedef std::vector<unsigned int> sequence_type;

std::ostream &operator<<(std::ostream &cout, sequence_type s);

vector<real_type> vlog(const vector<real_type> &u);

matrix<real_type> mlog(const matrix<real_type> &A);

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
	       real_type // the likelihood of the sequences of observations, under this new model
	       > learn(const std::vector<sequence_type> &obseqs);

  boost::tuple<HMM, // the learned model
	       real_type // the likelihood of the sequences of observations, under this new model
	       > baum_welch(const std::vector<sequence_type> &obseqs, //
			    real_type tolerance=1e-9, // tolerance level for convergence
			    unsigned int maxiter=200 // maximum number of iterations
			    );
};

boost::tuple<int,
	     real_type
	       > argmax(vector<real_type> &u);

std::ostream &operator<<(std::ostream &cout, HMM &hmm);

matrix<real_type> load_hmm_matrix(const char *filename);

vector<real_type> load_hmm_vector(const char *filename);

std::vector<sequence_type> load_hmm_observations(const char *filename);

