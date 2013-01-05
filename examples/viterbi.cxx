/*!
  \file viterbi.cxx
  \brief Example usage of the viterbi algorithm
  \author DOP (dohmatob elvis dopgima)
*/

#include "HiddenMarkovModelsNamespace.h" // pull-in in the entire API

using namespace HiddenMarkovModels; // so we don't get clobbered in the notations

/*!
  Main entry point.
*/
int main(int argc, char *argv[])
{
  // transition matrix
  RealMatrixType trans(2, 2);
  trans(0, 1) = trans(1, 0) = 0.6;
  trans(0, 0) = trans(1, 1) = 0.4;

  // emission matrix
  RealMatrixType em(2, 2);
  em(1, 0) = 0.41; em(1, 1) = 0.59;
  em(0, 1) = 0.33; em(0, 0) = 0.67;

  // initial distribution of hidden states
  RealVectorType pi(2);
  pi[0] = 0.75;
  pi[1] = 0.25;

  // initialize DiscreteHMM object
  DiscreteHMM dhmm(trans, em, pi);
  std::cout << "HMM:" << std::endl;
  std::cout << dhmm << std::endl;

  // prepare observations
  ObservationSequenceType obseq(6);
  obseq[0] = obseq[3] = obseq[5] = 1;
  obseq[1] = obseq[2] = obseq[4] = obseq[5] = 0;

  // run Viterbi algorithm
  HMMPathType trajectory = dhmm.viterbi(obseq);

  // render results
  std::cout << "The (a posteriori) most probable sequence of hidden states that generated the trace " << obseq << " is " << trajectory.get_path() << "." << std::endl;
  std::cout << "Its (log) likelihood is " << trajectory.get_likelihood() << ".\n" << std::endl;

  return 0;
}
