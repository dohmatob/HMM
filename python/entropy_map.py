"""
:Module: entropy_map
:Synopsis: Module contain core functions for doing entropic re-estimation of statistic parameters.
:Author: DOP (dohmatob elvis dopgima)

"""

__all__ = ['entropic_reestimate', '_BEAR',]

import sys
import os

import doctest
import unittest
import scipy.special.lambertw as W # Lambert's W function
from scipy import e, isreal
from numpy import sum, log, exp, abs, min, max, inf, nonzero, all, array, real, mean, spacing, setdiff1d

sys.path.append(os.path.dirname(sys.argv[0]))
from probability import normalize_probabilities, almost_uniform_vector
from convergence import check_converged

# constants
_EPS = spacing(1) # relative floating-point accuracy (distance beween 1.0 and next floating-point number) 
_BEAR = -2*log(_EPS) # exp(-x) < _EPS^2 (i.e near 0) for x > _BEAR

def entropic_reestimate(omega, theta=None, Z=1, maxiter=100, tol=1e-7, verbose=False):
    """
    Re-estimates a statistic parameter vector entropically [1]_.
    
    Parameters
    ----------
    omega : array_like 
        Evidence vector
    theta : array_like, optional
        Parameter vector to be re-estimated under given evidence and learning rate (default None)
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
    Z : float
        Final Learning rate
    _lambda : float
        Limiting value of Lagrange multiplier

    Examples
    --------
    >>> from entropy_map import entropic_reestimate
    >>> omega = [1, 2]
    >>> theta = [0.50023755, 0.49976245]
    >>> theta_hat, final_Z, _lambda = entropic_reestimate(omega, theta, Z=1, tol=1e-6)
    >>> theta_hat
    array([ 0.33116253,  0.66883747])
    >>> final_Z
    0.041828014112488016
    >>> _lambda
    -3.0152672618320637

    References
    ----------
    .. [1] Matthiew Brand, "Pattern learning via entropy maximization"

    """

    def _debug(msg=''):
        if verbose:
            print msg

    # XXX TODO: handle Z = 0 case
    assert Z != 0 

    # if no initial theta specified, start with uniform candidate
    if theta is None:
        theta = almost_uniform_vector(len(omega))

    # all arrays must be numpy-like
    omega = array(omega, dtype='float64')
    theta = array(theta, dtype='float64')

    # XXX TODO: trim-off any evidence which 'relatively close to 0' (since such evidence can't justify anything!) 
    pass

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
    converged = False
    _debug("entropy_map: starting Fixed-point loop ..\n")
    _debug("Initial model: %s"%theta)
    _debug("Initial lambda: %s"%_lambda)
    _debug("Initila learning rate (Z): %s"%Z)
    while not converged:
        # exhausted ?
        if maxiter <= iteration:
            break

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
        theta_hat = (-omega/Z)/g 
        assert all(theta_hat >= 0)

        # normalize the approximated theta_hat parameter
        theta_hat = normalize_probabilities(theta_hat)

        # re-estimate _lambda
        _lambda_hat = -(Z*(log(theta_hat[0]) + 1) + omega[0]/theta_hat[0]) # [0] or any other index [i]

        # compute relative gain in _lambda
        converged, _, relative_gain = check_converged(_lambda, _lambda_hat, tol=tol)

        # verbose for debugging, etc.
        _debug("Iteration: %d"%iteration)
        _debug('Current parameter estimate:\n%s'%theta)
        _debug('lambda: %s'%_lambda)
        _debug("Relative gain in lambda over last iteration: %s"%relative_gain)
        _debug("Learning rate (Z): %s"%Z)

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
    doctest.testmod(verbose=True)
    
