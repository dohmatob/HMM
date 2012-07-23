#ifndef DISCRETEHMM_H
#define DISCRETEHMM_H

/*!
  \file DiscreteHMM.h
  \brief Definition of DiscreteHMM class.
  \author DOP (dohmatob elvis dopgima)
  \warning I expect dynamically allocated/deallocated types like HiddenMarkovModels::HMMMatrixType, HiddenMarkovModels::HMMMVectorType, be wrapped in 'smart pointers' by client code (to avoid dangling pointers, etc.). I've tried my best not to corrupt such data whose reference is passed to my code (class constructors, etc.).

*/

#include "HMMBasicTypes.h"
#include "HMMPathType.h"

namespace HiddenMarkovModels
{
  /*!
    \class DiscreteHMM
    \brief Encapsulation of Hidden Markov Models with discrete emissions.
  */
  class DiscreteHMM 
  {
  private:
    /*! 
      Transition matrix of model .
    */
    HiddenMarkovModels::RealMatrixType _transition; 

    /*!
      Emission matrix of model.
    */
    HiddenMarkovModels::RealMatrixType _emission;

    /*!
      Initial distribution of hidden states of model.
    */
    HiddenMarkovModels::RealVectorType _pi;

  public:
    /*!
      Default constructor.

      \param transition probability matrix
      \param emission probability matrix
      \param pi initial probability vector
    */
    DiscreteHMM(HiddenMarkovModels::RealMatrixType& transition, 
		HiddenMarkovModels::RealMatrixType& emission,
		HiddenMarkovModels::RealVectorType& pi);

    /*!
      Set transition matrix of model.

      \param transition probability matrix
    */
    void set_transition(const HiddenMarkovModels::RealMatrixType& transition);

    /*!
      Set emission matrix of model.

      \param emission probability matrix
    */
    void set_emission(const HiddenMarkovModels::RealMatrixType& emission);

    /*!
      Set initial distribution of hidden states of mode.

      \param pi probability vector
    */
    void set_pi(const HiddenMarkovModels::RealVectorType& pi);

    /*!
      Retrieve number of hidden states in model.

      \return number of hidden states in model
    */
    int get_nstates() const;

    /*!
      Retrieve alphabet size of model (number of symbols that can be emitted).

      \return alphabet size of model
    */
    int get_nsymbols() const;

    /*!
      Retrieve transition matrix of model.

      \return transition matrix of model
    */
    const HiddenMarkovModels::RealMatrixType& get_transition() const;

    /*!
      Retrieve emission matrix of model.

      \return emission matrix of model
    */
    const HiddenMarkovModels::RealMatrixType& get_emission() const;

    /*!
      Retrieve initial distribution of hidden states of model.

      \return initial distribution of hidden states of model
    */
    const HiddenMarkovModels::RealVectorType& get_pi() const;

    /*! 
      Method to verify sanity of symbols (crucial, since symbols will be directly used as indices in emission matrix).
      
      \param i symbol to be checked for sanity
      \return boolean (symbol is sane or insane)
     **/
    bool is_symbol(unsigned int i) const;
    
    /*!
      The Viterbi algorithm.

      \param obseq the sequence of observations to be decoded
      \return a tuple of the optimal hidden state path to have produced the observation, and its likelihood
    */
    HiddenMarkovModels::HMMPathType viterbi(const HiddenMarkovModels::ObservationSequenceType& obseq);
    
    /*!
      Method to to compute forward and backward parameters of the Baum-Welch algorithm.
      
      \param obseq a sequence of observations       
      \return a tuple of (in technical jargon) alpha-hat, beta-hat, gamma-hat, and epsilon-hat
    */
    boost::tuple<HiddenMarkovModels::RealMatrixType, // alpha-hat
      HiddenMarkovModels::RealMatrixType, // beta-hat
      HiddenMarkovModels::RealMatrixType, // gamma-hat
      boost::multi_array< RealType, 3 >, // epsilon-hat
      HiddenMarkovModels::RealType // likelihood
      > forward_backward(const HiddenMarkovModels::ObservationSequenceType &obseq);
    
    /*! 
	Method to learn new model from old, using the Baum-Welch algorithm
	
	\param obseqs observation sequences to learn from
	\return a tuple of the learned model (HMM object) and the it likelihood
    */
    boost::tuple<HiddenMarkovModels::DiscreteHMM, 
      HiddenMarkovModels::RealType 
      > learn(const std::vector< HiddenMarkovModels::ObservationSequenceType > &obseqs);
    
    /*! 
      The Baum-Welch algorithm for multiple observation sequences.
      
      \param obseqs observation sequences to learn from
      \param tolerance tolerance level for convergence
      \param maxiter maximum number of iterations (-ve means boundless)
      \return likelihood of final model
    */
    HiddenMarkovModels::RealType baum_welch(const std::vector< HiddenMarkovModels::ObservationSequenceType > &obseqs, 
					    HiddenMarkovModels::RealType tolerance=1e-9, 
					    int maxiter=-1
					    );
  }; 
};

#endif // DISCRETEHMM_H
