/*!
  \file HMMIOUtils.cxx
  \brief Implementation of HMMIOUtil.h header file.
  \author DOP (dohmatob elvis dopgima)
*/

#include "HMMIOUtils.h"

std::ostream& HiddenMarkovModels::operator<<(std::ostream& cout, const HiddenMarkovModels::DiscreteHMM& dhmm)
{
  cout << "transition = " << dhmm.get_transition() << std::endl;
  cout << "emission = " << dhmm.get_emission() << std::endl;
  cout << "pi = " << dhmm.get_pi() << std::endl;

  return cout;
}
