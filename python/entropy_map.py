"""
:Module: entropy_map
:Synopsis: Module contain core functions for doing entropic re-estimation of statistic parameters.
:Author: DOP (dohmatob elvis dopgima)

"""

__all__ = ['entropic_reestimate', '_BEAR',]

import doctest
import unittest
import scipy.special.lambertw as W # Lambert's W function
from scipy import e, isreal
from numpy import sum, log, exp, abs, min, max, inf, nonzero, all, array, real, mean

import sys
import os
sys.path.append(os.path.dirname(sys.argv[0]))
from probability import normalize_probabilities, almost_uniform_vector

#constants
_BEAR = 100 # exp(-x) is near 0 for x > _BEAR

def entropic_reestimate(omega, theta, Z=1, maxiter=100, tol=1e-7, verbose=False):
    """
    Reeestimates a statistic parameter vector entropically [1]_.
    
    Parameters
    ----------
    omega : array_like 
        Evidence vector
    theta : array_like 
        Parameter vector to be re-estimated under given evidence and learning rate
    Z : {-1, 0, +1}, optional
        -1: Algorithm reduces to traditional MLE (e.g the Baum-Welch)

        0: ?

        +1: Algorithm will seek maximum structure
    maxiter : int, optional
        Maximum number of iterations of Fixed-point loop (default 100)
    verbose : bool, optional
        Display verbose output (default off)

    Returns
    -------
    theta_hat : array_like
        Learned parameter vector
    _lambda : float
        Langrange multiplier
    Z : float
        Final Learning rate

    Examples
    --------
    >>> from entropy_map import entropic_reestimate
    >>> omega = [1, 2]
    >>> theta = [0.50023755, 0.49976245]
    >>> theta_hat = entropic_reestimate(omega, theta)
    >>> theta_hat
    (array([ 0.33177538,  0.66822462]), 0.030109771542978479, -3.0109771551554658)

    References
    ----------
    .. [1] Matthiew Brand, "Pattern learning via entropy maximization"

    """

    def _debug(msg=''):
        if verbose:
            print msg

    assert Z != 0

    # all arrays must be numpy-like
    omega = array(omega, dtype='float64')
    theta = array(theta, dtype='float64')

    # prepare initial _lambda which will ensure that Lambert's W is real-valued
    if Z > 0:
        critical_lambda = min(-Z*(2 + log(omega/Z)))
        _lambda = critical_lambda - 1 # or anything less than the critical value above
    elif Z < 0:
        #  make an educated guess
        _lambda = -mean(Z*(log(theta) + 1) + omega/theta)
    assert all(-omega*exp(1+_lambda/Z)/Z > -1/e), -omega*exp(1+_lambda/Z)/Z 
    
    # Fixed-point loop
    theta_hat = theta
    iteration = 0
    relative_gain = inf
    _debug("entropy_map: starting Fixed-point loop ..\n")
    while relative_gain >= tol:
        # exhausted ?
        if maxiter < iteration:
            break

        _debug("Iteration: %d"%iteration)
        _debug('Current parameter estimate:\n%s'%theta)
        _debug('lambda: %s'%_lambda)
        _debug("Relative gain in lambda over last iteration: %s"%relative_gain)
        _debug("Learning rate (Z): %s"%Z)

        # if necessary, re-scale learning rate (Z) so that exp(1 + _lambda/Z) is not 'too small'
        if _lambda < 0:
            if Z > 0:
                new_Z = -_lambda/_BEAR
            elif Z < 0:
                new_Z = _lambda/_BEAR
            if new_Z != Z:
                Z = new_Z
                _debug("N.B:- We'll re-scale learning rate (Z) to %s to prevent Lambert's W function from vanishing."%(Z))

        # prepare argument (vector) for Lambert's W function
        z = -omega*exp(1 + _lambda/Z)/Z
        assert all(isreal(z)) 
        if any(z < -1/e):
            _debug("Lambert's W: argument z = %s out of range (-1/e, +inf)"%z)
            break

        # compute Lambert's W function at z
        if Z <= 0:
            g = W(z, k=0)
        else:
            g = W(z, k=-1)
        assert all(isreal(g))
        g = real(g)
        
        # check against division by zero (btw we re-scaled Z to prevent this)
        assert all(g != 0)

        # re-estimate theta
        theta_hat = (omega/Z)/g 

        # normalize the approximated theta_hat parameter
        theta_hat = normalize_probabilities(theta_hat)

        # re-estimate _lambda
        _lambda_hat = -(Z*(log(theta_hat[0]) + 1) + omega[0]/theta_hat[0]) # [0] or any other index [i]

        # compute relative gain in _lambda
        relative_gain = abs((_lambda - _lambda_hat)/_lambda)        

        # update _lambda and theta
        _lambda = _lambda_hat
        theta = theta_hat

        # goto next iteration
        iteration += 1

        _debug('\n')

    _debug("Done.")
    _debug('Final parameter estimate:\n%s'%theta)
    _debug('lambda: %s'%_lambda)
    _debug("Relative gain in lambda over last iteration: %s"%relative_gain)
    _debug("Learning rate (Z): %s"%Z)

    # converged ?
    if relative_gain < tol:
        _debug("entropic_reestimate: loop converged after %d iterations (tolerance was set to %s)"%(iteration,tol))
    else:
        _debug("entropic_reestimate: loop did not converge after %d iterations (tolerance was set to %s)"\
            %(maxiter,tol))

    # render results
    return theta_hat, Z, _lambda

if __name__ == '__main__':    
    omega = [1, 2, 3]
    theta = almost_uniform_vector(len(omega))
    theta_hat = entropic_reestimate(omega, theta, Z=1, maxiter=2)

    
