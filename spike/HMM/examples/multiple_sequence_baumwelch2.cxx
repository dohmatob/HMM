/*!
  \file multiple_sequence_baumwelch2.cxx 
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
  pi[0] = 31; pi[1] = 69;

  // transition matrix
  RealMatrixType trans(2, 2);
  trans(0, 0) = 4; trans(0, 1) = 6; // will be normalized to 0.4 0.6
  trans(1, 0) = 52; trans(1, 1) = 48; // will be normalized to 0.52 0.48

  // emission matrix
  RealMatrixType em(2, 2); // model emits two distrete symbols, 'a' and 'b'
  em(0, 0) = 49; em(0, 1) = 51; // will be normalized to 0.49 0.51
  em(1, 0) = 4; em(1, 1) = 6; // will be normalized to 0.4 0.6

  // initialize DiscreteHMM object
  DiscreteHMM dhmm(trans, em, pi);
  std::cout << "Initial model: " << std::endl << dhmm << std::endl;

  // prepare observation sequences
  std::vector < ObservationSequenceType > obseqs;
  ObservationSequenceType obseq;
  obseq += 1,1,1,0,0; // observation sequence 'bbbaa'
  obseqs += obseq; 
  obseq.clear();
  obseq += 1,0,1,1,0,0; // observation sequence 'babbaa'
  obseqs += obseq; 
  obseq.clear();
  obseq += 1,1,1,0,1,0,0; // observation sequence 'bbbabaa'
  obseqs += obseq; 
  obseq.clear();
  obseq += 1,1,0,1,1,0; // observation sequence 'bbabba'
  obseqs += obseq; 
  obseq.clear();
  obseq += 1,1,0,0; // observation sequence 'bbaa'
  obseqs += obseq; 
  obseq.clear();

  // run Baum-Welch algorithm
  dhmm.baum_welch(obseqs,
		  1e-15, // tolerance for convergence
		  1000 // maxiter
		  );

  return 0;
}
