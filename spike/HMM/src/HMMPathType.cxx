/*!
  \file HMMPathType.cxx
  \brief Implementation of HMMPathType.h header.
  \author DOP (dohmatob elvis dopgima)
*/

#include "HMMPathType.h"

HiddenMarkovModels::HMMPathType::HMMPathType(const HiddenMarkovModels::StateSequenceType& path,
					     HiddenMarkovModels::RealType likelihood)
{
  // set the path
  set_path(path);

  // set the path's likelihood
  set_likelihood(likelihood);
}

void  HiddenMarkovModels::HMMPathType::set_path(const HiddenMarkovModels::StateSequenceType& path)
{
  // XXX do some sanity checks of path before this
  _path = path;
}

void HiddenMarkovModels::HMMPathType::set_likelihood(HiddenMarkovModels::RealType likelihood)
{
  _likelihood = likelihood;
}

const HiddenMarkovModels::StateSequenceType& HiddenMarkovModels::HMMPathType::get_path() const
{
  return _path;
}

HiddenMarkovModels::RealType HiddenMarkovModels::HMMPathType::get_likelihood() const
{
  return _likelihood;
}

