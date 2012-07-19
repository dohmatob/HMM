/*!
  \file DiscreteHMM_constructor.cxx
  \brief So you see how DiscreteHMM objects can be created and manipulated.
  \author DOP (dohmatob elvis dopgima)
*/

#include "DiscreteHMM.h"
#include "HMMUtils.h"

using namespace HiddenMarkovModels;

/*!
  Entry point.
*/
int main(void)
{
  // local variables
  RealMatrixType A(2, 2); // transition matrix of model
  RealMatrixType B(2, 3); // emission matrix of model
  RealVectorType pi(2); // initial distribution of hidden states of model

  // initialize A
  A(1, 1) = A(0, 0) = 0.6;
  A(0, 1) = A(1, 0) = 0.4;

  // initialize B
  B(0, 0) = 0; B(0, 1) = B(0, 2) = 0.5;
  B(1, 0) = B(1, 1) = B(1, 2) = 1/3;

  // initialize pi
  pi[1] = 0.25;
  pi[0] = 0.75;

  // initialize DisceteHMM object
  DiscreteHMM dhmm(A, B, pi);

  // display Discrete hmm object
  std::cout << dhmm << std::endl;

  return 0;
}
