#ifndef PROGRESSBAR_H
#define PROGRESSBAR_H

/*!
  \file ProgressBar.h
  \brief Header file for ProgressBar class.
  \author DOP (dohmatob elvis dopgima)
*/

/*!
  \namespace BeautifulThings
  \brief Namespace for some routine beautification techno
*/
namespace BeautifulThings
{
  /*!
    \class ProgressBar
  */
  class ProgressBar
  {
  public:
    /*!
      Constructor.
      
      \param width width of progress bar
      \param resolution resolution of progress bar
    */
    ProgressBar(unsigned int width=1, unsigned int resolution=100);
    
    /*!
      Method to display a progress bar indicating the progress of the watched activity.
      
      \param completed quantity activity completed yet
      \param total total quantity of activity
      \param boomerang if true, fancy spinning bar will be displayed
    */
    void update(unsigned int completed, 
		unsigned int total,
		unsigned short boomerang=0);

    /*!
      Another version of update method.
    */
    void update();
    
    /*!
      Method to display a spinning bar.
    */
    void spin();

  private:
    /*!
      Width of progress bar.
    */
    unsigned int _width;

    /*!
      Resolution of progress bar.
    */
    unsigned int _resolution;
  };
};

#endif // PROGRESSBAR_H
