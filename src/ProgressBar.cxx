#include "ProgressBar.h"
#include <iostream>
#include <boost/assert.hpp>
#include <stdio.h>

BeautifulThings::ProgressBar::ProgressBar(unsigned int width, unsigned int resolution) : _width(width), _resolution(resolution)
{
}

/*!
  \warning Implement a GUI sensitive version of this function (Use Qt, OpenGL, etc. ?). It only works in console mode now.
*/
void BeautifulThings::ProgressBar::update(unsigned int completed, unsigned total, unsigned short boomerang)
{
  if(boomerang)
    {
      spin();
      return;
    }

  // only update _resolution times
  if(completed %(total/_resolution) != 0) 
    {
      return;
    }

  // calculate the ratio of completed-to-total
  float percent = 100*completed/(float)total;
  int   c     = percent * _width;
  
  // show the percentage complete
  printf("%3d%% [", (int)(percent));
  
  // show the load bar.
  for (int x = 0; x < c; x++)
    {
      std::cout << "=";
    }
 
  for (int x = c; x < _width; x++)
    {
      std::cout << " ";
    }
  
  // close bar
  if(percent == 100)
    {
      std::cout << "]";
    }
  
  // move to first column and clear current line
  std::cout << "\r"; 
  
  // flush stdout
  std::cout << std::flush;
}

void BeautifulThings::ProgressBar::spin()
{
  // local variables
  std::string slash[4] = {"|", "\\", "-", "/"}; // XXX this should be a static member of the class!!! No?
  static int x = 0;

  // update progress bar
  std::cout << "\r"; // carriage return back to beginning of line
  std::cout << "(busy here) " << slash[x] << " " << std::flush;
  
  // increment to make the slashs appears to rotatee
  x++;
  
  // reset slash animation
  if(x == 4)
    {
      x = 0;
    }
}

void BeautifulThings::ProgressBar::update()
{
  spin();
}










