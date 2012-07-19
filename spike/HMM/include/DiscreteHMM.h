#ifndef DISCRETEHMM_H
#define DISCRETEHMM_H

/*!
  \file DiscreteHMM.h
  \brief Definition of DiscreteHMM class.
  \author DOP (dohmatob elvis dopgima)
*/

#include "HMMBasicTypes.h"

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
    void set_transition(HiddenMarkovModels::RealMatrixType& transition);

    /*!
      Set emission matrix of model.

      \param emission probability matrix
    */
    void set_emission(HiddenMarkovModels::RealMatrixType& emission);

    /*!
      Set initial distribution of hidden states of mode.

      \param pi probability vector
    */
    void set_pi(HiddenMarkovModels::RealVectorType& pi);

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
  };
};

#endif // DISCRETEHMM_H
