import numpy

def normalize(x, dtype='float64'):
    assert x.ndim in [1,2] # vector or matrix

    x = x.astype(dtype)
    if x.ndim == 1:
        return x/x.sum()
    else:
        return x/numpy.array([x.sum(axis=1),]*x.shape[1]).T

def is_stochastic(x):
    assert x.ndim in [1, 2] # vector or matrix

    if x.ndim == 1:
        return numpy.all(x>=0)
    else:
        tmp = x.sum(axis=1)
        return numpy.all(tmp>=0)

def almost_uniform_vector(size):
    noise = numpy.random.normal(loc=0,scale=0.001,size=size)

    return normalize(numpy.ones(size, dtype='float64') + noise)

def almost_uniform_matrix(n, m=None):
    if m is None:
        m = n

    noise = numpy.random.normal(loc=0,scale=0.001,size=n*m)
    x = numpy.ones((n,m), dtype='float64')

    return normalize(x + noise.reshape(x.shape))
