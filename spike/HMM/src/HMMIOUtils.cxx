/*!
  \file HMMIOUtils.cxx
  \brief Implementation of HMMIOUtil.h header file.
  \author DOP (dohmatob elvis dopgima)
*/

#include "HMMIOUtils.h"
#include "ProgressBar.h"
#include <fstream>

std::ostream& HiddenMarkovModels::operator<<(std::ostream& cout, const HiddenMarkovModels::DiscreteHMM& dhmm)
{
  cout << "transition = " << dhmm.get_transition() << std::endl;
  cout << "emission = " << dhmm.get_emission() << std::endl;
  cout << "pi = " << dhmm.get_pi() << std::endl;

  return cout;
}

std::ostream &HiddenMarkovModels::operator<<(std::ostream &cout, const HiddenMarkovModels::ObservationSequenceType obseq)
{
  cout << "[" << obseq.size() << "](";
  for (int i = 0; i < obseq.size(); i++)
    {
      cout << obseq[i] << (i < obseq.size()-1 ? "," : "");
    }
  
  cout << ")";

  return cout;
}


HiddenMarkovModels::RealMatrixType HiddenMarkovModels::load_hmm_matrix(const char *filename)
{
  // XXX check that filename exists

  std::vector<std::vector<HiddenMarkovModels::RealType> > data;
  std::ifstream input(filename);
  std::string lineData;
  int n = 0;
  int m;

  while(std::getline(input, lineData))
    {
      HiddenMarkovModels::RealType d;
      std::vector<HiddenMarkovModels::RealType> row;
      std::stringstream lineStream(lineData);

      while (lineStream >> d)
	row.push_back(d);

      if (n == 0)
	{
	  m = row.size();
	}

      if (row.size() > 0)
	{
	  if (row.size() != m)
	    {
	      throw "mal-formed matrix line";
	    }

	  data.push_back(row);
	  n++;
	}
    }

  // pack the std array into a ublas matrix
  HiddenMarkovModels::RealMatrixType X(n, m);
  for (int i = 0; i < n; i++)
    {
      for (int j = 0; j < m; j++)
	{
	  X(i, j) = data[i][j];
	}
    }

  return X;
}

HiddenMarkovModels::RealVectorType HiddenMarkovModels::load_hmm_vector(const char *filename)
{
  // XXX check that filename exists

  return row(HiddenMarkovModels::load_hmm_matrix(filename), 0);
}

std::vector< HiddenMarkovModels::ObservationSequenceType > HiddenMarkovModels::load_hmm_observations(const char *filename)
{
  // XXX check that filename exists

  std::vector<HiddenMarkovModels::ObservationSequenceType> obseqs;
  std::ifstream input(filename);
  std::string lineData;
  BeautifulThings::ProgressBar progressbar;
  int wordcount = 0;

  progressbar.update();
  while(std::getline(input, lineData))
    {
      int d;
      HiddenMarkovModels::ObservationSequenceType row;
      std::stringstream lineStream(lineData);
      
      while (lineStream >> d)
	row.push_back(d);

      if (row.size() > 0)
	{
	  wordcount++;
	  obseqs.push_back(row);
	}

      progressbar.update();
    }

  return obseqs;
}

