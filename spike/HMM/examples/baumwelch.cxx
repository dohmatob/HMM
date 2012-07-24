/*!
  \file baumwelch.cxx 
  \brief Simple example of Baum-Welch application on single observation sequence
  \author DOP (dohmatob elvis dopgima)
*/

#include "HiddenMarkovModelsNamespace.h" 

using namespace HiddenMarkovModels;

/*!
  Main entry point.
*/
int main(int argc, char *argv[])
{
  // initial distribution of hidden states
  RealVectorType pi(3); // model has 3 hidden states, 0, 1, and 2
  pi[0] = 0.5; pi[1] = 0.3; pi[2] = 0.2;

  // transition matrix
  RealMatrixType trans(3, 3);
  trans(0, 0) = 0.45; trans(0, 1) = 0.35; trans(0, 2) = 0.20;
  trans(1, 0) = 0.10; trans(1, 1) = 0.50; trans(1, 2) = 0.40;
  trans(2, 0) = 0.15; trans(2, 1) = 0.25; trans(2, 2) = 0.60;

  // emission matrix
  RealMatrixType em(3, 2); // model emits two distrete symbols, 'a' and 'b'
  em(0, 0) = em(2, 1) = 1;
  em(0, 1) = em(2, 0) = 0;
  em(1, 0) = em(1, 1) = 0.5;

  // initialize DiscreteHMM object
  DiscreteHMM dhmm(trans, em, pi);
  std::cout << "Initial model: " << std::endl << dhmm << std::endl;

  // prepare observation sequences
  std::vector < ObservationSequenceType > obseqs;
  ObservationSequenceType obseq;
  obseq += 0,1,1,0,0; // obseq = observation sequence 'abbaa'
  obseqs += obseq; // obseqs = pool of 1 observation sequence 'abbaa'

  // run Baum-Welch algorithm
  dhmm.baumwelch(obseqs,
		  1e-15, // tolerance for convergence
		  150 // maxiter
		  );

  return 0;
}
