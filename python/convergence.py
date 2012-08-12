"""
:Module: convergence
:Synopsis: Module exporting functions and constants related to convergence (in the spirit of Cauchy)
:Author: DOHMATOB Elvis Dopgima
"""

__all__ = ['compute_relative_error', 'check_converged']

from numpy import spacing, abs

# constants
_EPS = spacing(1)

def compute_relative_error(previous, current):
    """
    Calculates the relative error of one value over another.

    Parameters
    ----------
    previous : real
        Previous value.
    current : real
        Current value.

    Returns
    -------
    relative_error : real
        The relative error of current value over previous value.
    increased : bool
        
        1: The values have increased.

        0: The values have decreased.

     Examples
     --------
     >>> from convergence import compute_relative_error
     >>> relative_error, increased = compute_relative_error(0.68, 0.5967)
     >>> relative_error
     0.13049267643142481
     >>> increased
     False
     >>> relative_error, increased = compute_relative_error(0.682, 0.6905)
     >>> relative_error
     0.012386156648451658
     >>> increased
     True
     
     """
    
    # did values increase ?
    increased = (current >= previous)


    # compute average value in the change previous -> current
    avg = (abs(current) + abs(previous) + _EPS)/2

    # compute relative error
    relative_error = abs(current - previous)/avg

    # render results
    return relative_error, increased
    

def check_converged(previous, current, tol=1e-4):
    """
    Function to check whether a sequence of real values has converged w.r.T. a prescribed tolerance.

    Parameters
    ----------
    previous : real
        Previous value of sequence.
    current : real
        Current value of sequence.
    tol : pisitive real, optional (default 1e-4)
        Tolerance for convergence test.

    Returns
    -------
    converged : bool
        
        1: The values have converged.

        0: The values have not converged.
    increased : bool
        
        1: The values have increased.

        0: The values have decreased.

    Examples
    --------
    >>> from convergence import check_converged
    >>> converged, increased, _ = check_converged(-17234.89, -17234.72345, tol=1e-4)
    >>> converged
    True
    >>> increased
    True
    >>> converged, increased, _ = check_converged(-17234.89, -17234.72345, tol=1e-6)
    >>> converged
    False
    >>> increased
    True

    """

    # compute relative error
    relative_error, increased = compute_relative_error(previous, current)

    # did values converged ?
    converged = relative_error < tol

    # render results
    return converged, increased, relative_error
    
