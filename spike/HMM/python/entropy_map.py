# (c) DOP (dohmato elvis dopgima)

import scipy.special.lambertw as W
from scipy import e, isreal
from numpy import sum, log, exp, abs, array, min, max

def normalize_probabilities(x):
    return x/sum(x, dtype='float64')

def entropy_map(omega, theta, nu=-1, _lambda=None, tol=1e-3):
    if _lambda is None:
        _lambda = (2/nu+log(-nu*max(omega/theta))/nu) - 1 # or anythx less than (2/nu+log(-nu*max(omega/theta))/nu)
        
    L = len(omega)
    theta_hat = theta

    # Fixed-point loop
    iteration = 0
    while True:
        print "Iteration: %d"%iteration
        print "\tlambda: %s"%_lambda
        for i in xrange(L):
            z = nu*omega[i]*exp(1-nu*_lambda)/theta[i]
            assert -1 <= e*z < 0, "Lambert's W: arg %s out of range"%z
            g = W(z, k=-1) # lower-branch of W on -1/e <= z < 0 verifying W <= -1, denoted W_(-1)
            assert isreal(g)

            # compute theta_hat[i] given _lambda
            theta_hat[i] = nu*omega[i]/g 

        # normalize the approximated theta_hat parameter
        theta_hat = normalize_probabilities(theta_hat)

        # re-estimate _lambda
        _lambda_hat = -omega[0]/theta_hat[0] + log(theta_hat[0]/theta[0])/nu + 1/nu

        # compute relative gain in _lambda
        relative_gain = abs((_lambda - _lambda_hat)/_lambda)        
        print "\tRelative gain in lambda over last iteration: %s"%relative_gain

        # converged ?
        if relative_gain < tol:
            break

        _lambda = _lambda_hat
        theta = theta_hat

        # goto next iteration
        iteration += 1
    
    return theta_hat

if __name__ == '__main__':
    print entropy_map(array([1,2,3]), array([0.6,0.08,0.32]))
    
