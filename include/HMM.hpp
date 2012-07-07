#ifndef HMM_H
#define HMM_H

// Boost 
#include <boost/numeric/ublas/matrix.hpp> // matrix algebra
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
  /**
   * basic types
   **/
  typedef double real_type;
  typedef std::vector<unsigned int> sequence_type;
  typedef ublas::matrix<real_type> matrix;
  typedef ublas::vector<real_type> vector;

  /** 
      Function to display sequences.
   **/
  std::ostream &operator<<(std::ostream &cout, sequence_type seq);

  /**
     Function to compute logarithm of vector.

     @param u - vector whose logarithm is to be computed
     @return Logarithm of input vector
  **/
  vector vlog(const vector &u);
  
  /** 
      Function to compute logarithm of matrix.
      
      @param m - matrix whose logarithm is to be computed
      @return logarithm of input matrix
  **/
  matrix mlog(const matrix &A);

  /**
     Function to check whether matrix is stochastic.

     @param m - matrix to be checked for stochasticity
     @return true if matrix if stochastic, false otherwise
  **/
  bool is_stochastic_matrix(const matrix& m);

  /**
     Function to check whether vector is stochastic.
     
     @param v - vector to be checked for stochasticity
     @return true if vector if stochastic, false otherwise
  **/
  bool is_stochastic_vector(const vector& v);

  /** @brief Class to incapsulate Hidden Markov Models
      
      @author DOP (dohmatob elvis dopgima)
  **/
  class HMM
  {
  private:
    int _nstates;/**<number of hidden states in the the model*/
    int _nsymbols; /**<size of observable alphabet*/
    matrix _transition; /**<transition probabilities*/
    matrix _emission; /**<emission probabilites*/
    vector _pi; /**<initial distribution of hidden states*/
    
  public:
    /**
       Default constructor.

       @param transition - transition matrix of model
       @param emission - emission matrix of model 
       @param pi - initial distribution of hidden states of model
       @return handle to newly instantiated HMM object
     **/
    HMM(matrix transition,
	matrix emission,
	vector pi);

    /**
       Constructor by model order.

       @param nstates - the number of hidden states in the model
       @param nsymbols - the emitted symbol alphabet size of the model
    **/
    HMM(int nstates, int nsymbols);
    
    /**
       Method to set model transition matrix of model.

       @param transition - transition matrix to assign to model
    **/
    void set_transition(const matrix& transition);
    
    /**
       Method to set model emission matrix of model.
       
       @param emission - emission matrix to assign to model
    **/
    void set_emission(const matrix& emission);
    
    /**
       Method to set initial distribution of hidden states of model.
       
       @param pi - initial distribution of hidden states, to assign to model
    **/
    void set_pi(const vector& pi);

    /**
       Method to get model number of hidden states.

       @return number of hidden states of model
     **/
    int get_nstates(void);

    /**
       Method to get model number of symbols.

       @return size of symbol alphabet of model
     **/
    int get_nsymbols(void);

    /**
       Method to get model transition matrix.

       @return transition matrix of model
     **/
    const matrix &get_transition(void);

    /**
       Method to get model emission matrix.

       @return emission matrix of model
     **/
    const matrix &get_emission(void);

    /**
       Method to get model initial distribution of hidden states.
       
       @return initial distribution of hidden states of model
     **/
    const vector &get_pi(void);
    
    /** 
	Method to verify sanity of symbols (crucial, since symbols will be directly used as indices in emission matrix).
	
	@param i - symbol to be checked for sanity
	@return boolean (symbol is sane or insane)
     **/
    bool is_symbol(unsigned int i);
    
    /** 
	The Viterbi algorithm.

	@param obseq - the sequence of observations to be decoded
	@return a tuple of the optimal hidden state path to have produced the observation, and its likelihood
     **/
    boost::tuple<sequence_type, // optimal path
      real_type // likelihood of path
      > viterbi(const sequence_type &obseq);
    
    /**
       Method to to compute forward and backward parameters of the Baum-Welch algorithm.

       @param obseq - a sequence of observations       
       @return a tuple of (in technical jargon) alpha-hat, beta-hat, gamma-hat, and epsilon-hat
     **/
    boost::tuple<matrix, // alpha-hat
		 matrix, // beta-hat
		 matrix, // gamma-hat
		 boost::multi_array<real_type, 3>, // epsilon-hat
		 real_type // likelihood
		 > forward_backward(const sequence_type &obseq);
  
    /** 
	Method to learn new model from old, using the Baum-Welch algorithm
	
	@param obseqs - observation sequences to learn from
	@return a tuple of the learned model (HMM object) and the it likelihood
    **/
    boost::tuple<HMM, 
		 real_type 
		 > learn(const std::vector<sequence_type> &obseqs);
  
    /** 
	The Baum-Welch algorithm for multiple observation sequences.

	@param obseqs - observation sequences to learn from
	@param tolerance - tolerance level for convergence
	@param maxiter - maximum number of iterations
	@return a tuple of the learned model (HMM object) and the it likelihood
     **/
    boost::tuple<HMM,
		 real_type 
		 > baum_welch(const std::vector<sequence_type> &obseqs, 
			      real_type tolerance=1e-9, 
			      unsigned int maxiter=200 
			      );
  }; 
  
  /**
     Function to compute the maximum value of a vector and the (first!)  index at which it is attained.

     @param u - vector whose argmax is sought-for
     @return - a tuple of the index of which the max is attained, and the max itself
   **/
  boost::tuple<int,
	       real_type
	       > argmax(vector &u);
  
  /**
     An overloading of the std operator <<, So we may display HMM objects.

     @param cout - cout
     @param hmm - HMM object to be displayed
   **/
  std::ostream &operator<<(std::ostream &cout, HMM &hmm);
 
  /**
     Function to load a matrix from a file.

     @param filename - name of file containing matrix to be read
     @return - read matrix
   **/
  matrix load_hmm_matrix(const char *filename);
  
  /** 
      Function to load a vector from a file.
   **/
  vector load_hmm_vector(const char *filename);
  
  /** 
   * function to load a sequence of observations from a file
   **/
  std::vector<sequence_type> load_hmm_observations(const char *filename);
}

#endif // HMM_H
