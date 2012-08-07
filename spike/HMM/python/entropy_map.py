# (c) DOP (dohmato elvis dopgima)
#
# Inspired by Matthiew Brand's "Pattern learning via entropy maximization"

import scipy.special.lambertw as W
from scipy import e, isreal
from numpy import sum, log, exp, abs, min, max, inf, nonzero, all, array, real, mean

# constants
_BEAR = -50 # exp(-x) is near 0 for x > _BEAR

def normalize_probabilities(x):
    """
    Function to normalize a probability vector so it becomes stochasitc
    """
    return x/sum(x, dtype='float64')

def entropic_estimate(omega, 
                      theta, 
                      Z=1, # relative importance of entropy over likelihood
                      maxiter=100, 
                      tol=1e-7,
                      ):
    """
    Re-estimates a parameter (statistic) theta = (theta[0], theta[1], ...) based on evidence vector omega.
    XXX omega should contain no zero evidence and theta should have no zero coordinates (if these conditions fail,
    then simply trim off the offending parameters!)
    """
    
    assert Z != 0

    # all arrays must be numpy-like
    omega = array(omega)
    theta = array(theta)

    # prepare initial _lambda which will ensure that Lambert's W is real-valued
    if Z > 0:
        critical_lambda = min(-Z*(2 + log(omega/Z)))
        _lambda = critical_lambda - 1 # or anything less than the critical value above
    elif Z < 0:
        _lambda = -mean(Z*(log(theta) + 1) + omega/theta)
    assert all(-omega*exp(1+_lambda/Z)/Z > -1/e), -omega*exp(1+_lambda/Z)/Z 
    
    # Fixed-point loop
    theta_hat = theta
    iteration = 0
    relative_gain = inf
    print "entropy_map: starting Fixed-point loop .."
    print
    while relative_gain >= tol:
        # exhausted ?
        if maxiter < iteration:
            break

        print "Iteration: %d"%iteration
        print 'Current parameter estimate:\n%s'%theta
        print 'lambda:', _lambda
        print "Relative gain in lambda over last iteration: %s"%relative_gain

        # if necessary, re-scale learning rate (Z) so that exp(1 + _lambda/Z) is not 'too small'
        if _lambda < 0:
            if Z > 0:
                new_Z = max([Z, -_lambda/_BEAR])
            elif Z < 0:
                new_Z = min([Z, -_lambda/_BEAR])
            if Z != new_Z:
                Z = new_Z
                print "N.B: Re-scaled learning rate (Z) to %s so Lambert's W function doesn't vanish."%Z

        # prepare argument for Lambert's W function
        z = -omega*exp(1 + _lambda/Z)/Z
        assert all(isreal(z)) 
        if any(z < -1/e):
            print "Lambert's W: arg %s out of range"%z
            break

        # compute Lambert's W function at z
        if Z <= 0:
            g = W(z, k=0)
        else:
            g = W(z, k=-1)
        assert all(isreal(g))
        g = real(g)
        
        # check against division by zero
        if any(g == 0):
            print 'Fatal!'
            break

        # re-estimate theta
        theta_hat = (-omega/Z)/g 

        # normalize the approximated theta_hat parameter
        theta_hat = normalize_probabilities(theta_hat)

        # re-estimate _lambda
        _lambda_hat = -(Z*(log(theta_hat[0]) + 1) + omega[0]/theta_hat[0])

        # compute relative gain in _lambda
        relative_gain = abs((_lambda - _lambda_hat)/_lambda)        

        # update _lambda and theta
        _lambda = _lambda_hat
        theta = theta_hat

        # goto next iteration
        iteration += 1

        print

    print "Done."

    # converged ?
    if relative_gain < tol:
        print "entropic_estimate: loop converged after %d iterations (tolerance was set to %s)"%(iteration,tol)
    else:
        print "entropic_estimate: loop did not converge after %d iterations (tolerance was set to %s)"\
            %(maxiter,tol)

    print "Final learning rate (Z):", Z    
    print "Final parameters:\n%s"%theta_hat

    # render results
    return theta_hat, Z, _lambda

if __name__ == '__main__':
    from hmm import almost_uniform_vector
    
    omega = [1, 2]
    theta = almost_uniform_vector(len(omega))
    theta_hat = entropic_estimate(omega, theta, Z=1)

    
