/*!
  \file multiple_sequence_baumwelch.cxx 
  \brief Simple example of Baum-Welch application for multiple observation sequences
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
  RealVectorType pi(2); // model has 3 hidden states, 0, 1
  pi[0] = 0.31; pi[1] = 0.69;

  // transition matrix
  RealMatrixType trans(2, 2);
  trans(0, 0) = 0.4; trans(0, 1) = 0.6; 
  trans(1, 0) = 0.52; trans(1, 1) = 0.48; 

  // emission matrix
  RealMatrixType em(2, 2); // model emits two distrete symbols, 'a' and 'b'
  em(0, 0) = 0.49; em(0, 1) = 0.51; 
  em(1, 0) = 0.4; em(1, 1) = 0.6; 

  // initialize DiscreteHMM object
  DiscreteHMM dhmm(trans, em, pi);
  std::cout << "Initial model: " << std::endl << dhmm << std::endl << std::endl;

  // prepare observation sequences
  std::vector < ObservationSequenceType > obseqs; 
  ObservationSequenceType obseq; // holds a single observation sequence
  obseq += 0,0,0,1,1; // observation sequence 'aaabb'
  obseqs += obseq; 
  obseq.clear();
  obseq += 0,1,0,0,1,1,1; // observation sequence 'abaabbb'
  obseqs += obseq; 
  obseq.clear();
  obseq += 0,0,0,1,0,1,1; // observation sequence 'aaababb'
  obseqs += obseq; 
  obseq.clear();
  obseq += 0,0,1,0,1; // observation sequence 'aabab'
  obseqs += obseq; 
  obseq.clear();
  obseq += 0,1; // observation sequence 'ab'
  obseqs += obseq; 
  obseq.clear();

  // run Baum-Welch algorithm
  dhmm.baumwelch(obseqs,
		  1e-15, // tolerance for convergence
		  1000 // maxiter
		  );

  return 0;
}
