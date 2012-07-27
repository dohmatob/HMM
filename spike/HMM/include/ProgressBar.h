#ifndef PROGRESSBAR_H
#define PROGRESSBAR_H

/*!
  \file ProgressBar.h
  \brief Header file for ProgressBar class.
  \author DOP (dohmatob elvis dopgima)
*/

#include <sstream> // for 'std::stringstream' type

/*!
  \namespace BeautifulThings
  \brief Namespace for some routine beautification techno
*/
namespace BeautifulThings
{
  class ProgressBar
  {
  public:
    /*!
      Constructor.
      
      \param slider indicates that bar shall slide along as it spins
    */
    ProgressBar(unsigned int _width=1, unsigned int _resolution=100);
    
    /*!
      Method to display a spinning bar to indicate the progress of the watched activity.
      
      \param percent the percentage of the watched activity completed yet
    */
    void update(unsigned int completed, 
		unsigned int total,
		unsigned short boomerange=0);

    void update();
    
    void spin();

  private:
    unsigned int _width;

    unsigned int _resolution;

    std::stringstream _bars;

    int _completed; 
  };
};

#endif // PROGRESSBAR_H
