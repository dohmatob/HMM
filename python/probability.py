"""
:Module: probability
:Synopsis: Module for routine stuff related to probability distributions, etc.
:Author: DOP (dohmatob elvis dopgima)

"""

__all__ = ['normalize_probabilities', 'is_stochastic', 'almost_uniform_vector', 'almost_uniform_matrix']

import doctest
from numpy import sum, all, array, ones
from numpy.random import normal

def normalize_probabilities(x, dtype='float64'):
    """
    Function to normalize a probability vector so it becomes stochasitc.
    
    Parameters
    ----------
    x : array_like
        probabilities to normalize
    dtype : numpy dtype_like, optional
        data type of the (output) probability terms (default 'float64')
    
    Returns
    -------
    y : array_like
        normalized probabilities

    Examples
    --------
    >>> from probability import normalize_probabilities
    >>> from numpy import sum
    >>> normalize_probabilities([1, 2, 3])
    array([ 0.16666667,  0.33333333,  0.5       ])
    >>> sum(_)
    1.0

    """
    if x.ndim == 1:
        return x/sum(x, dtype='float64')
    elif x.ndim == 2:
        return x/array([sum(x, dtype='float64', axis=1),]*x.shape[1]).T

def is_stochastic(x):
    assert x.ndim in [1, 2] # vector or matrix

    if x.ndim == 1:
        return all(x>=0)
    else:
        tmp = x.sum(axis=1)
        return all(tmp>=0)

def almost_uniform_vector(size):
    noise = normal(loc=0,scale=0.001,size=size)

    return normalize_probabilities(ones(size, dtype='float64') + noise)

def almost_uniform_matrix(n, m=None):
    if m is None:
        m = n

    noise = normal(loc=0,scale=0.001,size=n*m)
    x = ones((n,m), dtype='float64')

    return normalize_probabilities(x + noise.reshape(x.shape))

def test_normalize_uniform_probabilities():
    a = array([1, 1, 1])
    u = normalize_probabilities(a) 
    assert all(3*u == 1)

def test_normalize_nonuniform_probabilities():
    a = array([3, 0, 3])
    u = normalize_probabilities(a) 
    assert u[0] == u[2] == 0.5 == 0.5 - u[1]

if __name__ == '__main__':
    doctest.testmod()
