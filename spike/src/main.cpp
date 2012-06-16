// (c) 2012 DOP (dohmatob elvis dopgima)
// main.cpp: main source file

#include "HMM.h"
#include <algorithm> // random_shuffle, etc.
#include <ctype.h> //__toascii
#include <stdio.h> // printf

int main(void)
{
  // XXX refactor main into unittest cases
  std::cout << "Loadin: HMM parameters from files .." << std::endl;
  matrix<real_type> trans = load_hmm_matrix("corpus_transition.dat");
  matrix<real_type> em = load_hmm_matrix("corpus_emission.dat");
  vector<real_type> pi = load_hmm_vector("corpus_pi.dat");
  std::cout << "Done.\n" << std::endl;

  // initialize HMM object
  HMM hmm(trans,em,pi);
  std::cout << "HMM:\n" << hmm;

  // prepare data
  std::cout << "Loadin: english words from corpus file .." << std::endl;
  std::vector<sequence_type> corpus = load_hmm_observations("corpus_words.dat"); // load
  std::cout << "Done (loaded " << corpus.size() << " words).\n" << std::endl;


  // draw a random sample
  int nlessons = 1000;
  std::cout << "Sampling " << nlessons << " words from corpus .." << std::endl;
  std::random_shuffle(corpus.begin(), corpus.end());
  std::vector<sequence_type> lessons = std::vector<sequence_type>(corpus.begin(),corpus.begin()+nlessons%corpus.size());
  std::cout << "Done.\n" << std::endl;

  boost::tuple<sequence_type,
	       real_type
	       > path = hmm.viterbi(lessons[2]);

  std::cout << "The a posteriori most probable sequence of hidden states that generated the trace " << lessons[2] << " is " << boost::get<0>(path) << "." << std::endl;
  std::cout << "Its (log) likelihood is " << boost::get<1>(path) << ".\n" << std::endl;

  // Bauw-Welch
  hmm.baum_welch(lessons);
  std::cout << "\nFinal HMM:\n" << hmm;

  std::cout << "Viterbi classification of the 26 symbols (cf. letters of the english alphabet):" << std::endl;
  sequence_type symbol(1);
  unsigned int correction;
  for (int j = 0; j < 26; j++)
    {
      symbol[0] = j;
      boost::tuple<sequence_type,
		   real_type
		   > path = hmm.viterbi(symbol);

      unsigned int which = boost::get<0>(path)[0]; // vowel or consonant ?

      // let's call a's cluster "vowel" and call the other cluster "consonant"
      if (j == 0)
	{
	  correction = which;
	}
      which = correction ? 1 - which : which;

      printf("\t%c is a %s\n", __toascii('A')+j,which?"consonant":"vowel");
      // std::cout << "\t" << j << " is in class " << boost::get<0>(path)[0] << std::endl;
    }

  return 0;
}
