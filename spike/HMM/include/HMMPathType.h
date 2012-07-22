#ifndef HMMPATHTYPE_H
#define HMMPATHTYPE_H

/*!
  \file HMMPathType.h
  \brief Header file for HMMPathType class.
  \author DOP (dohmatob elvis dopgima)
*/

#include "HMMBasicTypes.h"

namespace HiddenMarkovModels
{
  /*!
    \class HMMPathType
    \brief Encapsulates hidden path traversed, together with the path's likelihood amongst other possible paths.
  */
  class HMMPathType
  {
  private:
    /*!
     Traversed path.
    */
    HiddenMarkovModels::StateSequenceType _path;
    
    /*!
      Likelihood of path.
    */
    HiddenMarkovModels::RealType _likelihood;

  public:
    /*!
      Constructor.

      \param path traversed path
      \param likelihood likelihood of path
    */
    HMMPathType(const HiddenMarkovModels::StateSequenceType& path,
		HiddenMarkovModels::RealType likelihood);
    
    /*!
      Method to set path.

      \param path path
    */
    void set_path(const HiddenMarkovModels::StateSequenceType& path);

    /*!
      Method to set likelihood.
      
      \param likelihood likelihood
    */
    void set_likelihood(HiddenMarkovModels::RealType likelihood);

    /*!
      Method to get path.

      \return path
    */
    const HiddenMarkovModels::StateSequenceType& get_path() const;

    /*!
      MEthod to get likelihood of path.

      \return likelihood of path
    */
    HiddenMarkovModels::RealType get_likelihood() const;
  }; // class HMMPathType
}; // namespace HiddenMarkovModels

#endif // HMMPATHTYPE_H
