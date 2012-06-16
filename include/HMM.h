// (c) 2012 DOP (dohmatob elvis dopgima)
// HMM.h: principal header file

// Boost 
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp> // extract matrix row, matrix column, etc.
#include <boost/numeric/ublas/io.hpp> // display matrix, etc.
#include <boost/multi_array.hpp> // for multi-dimensional arrays (aka tensors), etc.
#include <boost/assign/std/vector.hpp> // operator+=() for vectors, etc.
#include <boost/assert.hpp>
#include <boost/tuple/tuple.hpp> // so I can return multiple values from functions (like in python)

// native headers
#include <math.h> // log, abs, etc.
#include <iostream> // cout, etc.
#include <fstream> // file handling functions like getline, etc.
#include <sstream>

namespace ublas = boost::numeric::ublas; // namespace alias (saner than 'using ...')
using namespace boost::assign;


namespace HiddenMarkovModels
{
  // basic types
  typedef double real_type;
  typedef std::vector<unsigned int> sequence_type;

  // so we can display sequences
  std::ostream &operator<<(std::ostream &cout, sequence_type s);

  // method to compute logarithm of vector
  ublas::vector<real_type> vlog(const ublas::vector<real_type> &u);
 
  // method to compute logarithm of matrix
  ublas::matrix<real_type> mlog(const ublas::matrix<real_type> &A);
  
  // And now, the real stuff
  class HMM
  {
  private:
    int _nstates; // number of hidden states in the the model
    int _nsymbols; // size of observable alphabet
    ublas::matrix<real_type> _transition; // transition probabilities
    ublas::matrix<real_type> _emission; // emission probabilites
    ublas::vector<real_type> _pi; // initial distribution of hidden states
    
  public:
    HMM(ublas::matrix<real_type> transition,
	ublas::matrix<real_type> emission,
	ublas::vector<real_type> pi);
    
    // methods to get model parameters
    int get_nstates(void);
    int get_nsymbols(void);
    const ublas::matrix<real_type> &get_transition(void);
    const ublas::matrix<real_type> &get_emission(void);
    const ublas::vector<real_type> &get_pi(void);
    
    // method to verify sanity of symbols (crucial, since symbols will be directly used as indices in emission matrix)
    bool isSymbol(unsigned int i);
    
    // the Viterbi algorithm
    boost::tuple<sequence_type, // optimal path
      real_type // likelihood of path
      > viterbi(const sequence_type &obseq);
    
    // method to to compute forward and backward parameters of the Baum-Welch algorithm
    boost::tuple<ublas::matrix<real_type>, // alpha-hat
      ublas::matrix<real_type>, // beta-hat
      ublas::matrix<real_type>, // gamma-hat
      boost::multi_array<real_type,3>, // epsilon-hat
      real_type // likelihood
      > forward_backward(const sequence_type &obseq);
    
    // method to learn new model from old, using the Baum-Welch algorithm
    boost::tuple<HMM, // the learned model
      real_type // the likelihood of the sequences of observations, under this new model
      > learn(const std::vector<sequence_type> &obseqs);
    
    // the Baum-Welch algorithm for multiple observation sequences
    boost::tuple<HMM, // the learned model
      real_type // the likelihood of the sequences of observations, under this new model
      > baum_welch(const std::vector<sequence_type> &obseqs, //
		   real_type tolerance=1e-9, // tolerance level for convergence
			    unsigned int maxiter=200 // maximum number of iterations
		   );
  }; // HMM
  
  // function to compute the maximum value of a vector and the (first!)  index at which it is attained 
  boost::tuple<int,
    real_type
    > argmax(ublas::vector<real_type> &u);
  
  // so we may display HMM objects
  std::ostream &operator<<(std::ostream &cout, HMM &hmm);
 
  // function to load an matrix from a file
  ublas::matrix<real_type> load_hmm_matrix(const char *filename);
 
  // function to load a vector from a file
  ublas::vector<real_type> load_hmm_vector(const char *filename);
  
  // function to load a sequence of observations from a file
  std::vector<sequence_type> load_hmm_observations(const char *filename);
}
