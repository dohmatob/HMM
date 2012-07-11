#ifndef HMM_H
#define HMM_H

/*!
  \file HMM.hpp
  \brief Specification of HiddenMarkovModels interface.
  \author DOP (dophmatob elvis dopgima)
*/

#include "HMMBasicTypes.hpp"

// native headers
#include <math.h> // log, abs, etc.
#include <iostream> // cout, etc.
#include <fstream> // file handling functions like getline, etc.
#include <sstream>

/*!
  \namespace HiddenMarkovModels
  \brief This namespace groups types (classes, etc.) and functions related to Hidden Markov Models.
  \author DOP (dohmatob elvis dopgima)
*/
namespace HiddenMarkovModels
{
  /*!
    \class HMM
    \brief This class declares the structure of a Hidden Markov Model object
  */
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
    int get_nstates(void) const;

    /**
       Method to get model number of symbols.

       @return size of symbol alphabet of model
     **/
    int get_nsymbols(void) const;

    /**
       Method to get model transition matrix.

       @return transition matrix of model
     **/
    const matrix &get_transition(void) const;

    /**
       Method to get model emission matrix.

       @return emission matrix of model
     **/
    const matrix &get_emission(void) const;

    /**
       Method to get model initial distribution of hidden states.
       
       @return initial distribution of hidden states of model
     **/
    const vector &get_pi(void) const;
    
    /** 
	Method to verify sanity of symbols (crucial, since symbols will be directly used as indices in emission matrix).
	
	@param i - symbol to be checked for sanity
	@return boolean (symbol is sane or insane)
     **/
    bool is_symbol(unsigned int i) const;
    
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
