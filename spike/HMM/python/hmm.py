# (C) DOP (dohmatob elvis dopgima)
# Python module implementing Hidden Markov Models (discrete)
#
# Caution: This is just spiky code!!!
#
# TODO: 
#     - Refactor the code
#     - Documentation with sphinx
#     - Profiling 
#     - Cython optimization

import numpy
import unittest
import os

import sys
sys.path.append(os.path.dirname(sys.argv[0]))
from probability import normalize, almost_uniform_matrix, almost_uniform_vector, is_stochastic
    

def chopper(filename):
    ifh = open(filename)
    while 0x1:
        line = ifh.readline()
        if not line:
            break
        yield [int(item) for item in line.rstrip('\r\n').split(' ')]
    # ifh.close()


class DiscreteHMM:
    _REALTYPE = 'float64'

    def __init__(self, nstates=None, nsymbols=None, transition=None, emission=None, pi=None):
        assert not ((nstates is None and transition is None and pi is None) or (nsymbols is None and emission is None))

        if not transition is None:
            assert transition.ndim == 2
            if not nstates is None:
                assert nstates == transition.shape[0]
            else:
                nstates = transition.shape[0]
            if not emission is None:
                assert transition.shape[0] == emission.shape[0]
            if not pi is None:
                assert transition.shape[0] == pi.size
            
        if not emission is None:
            assert emission.ndim == 2
            if not nsymbols is None:
                assert nsymbols == emission.shape[1]
            else:
                nsymbols = emission.shape[1]
            if not nstates is None:
                assert nstates == emission.shape[0]
            else:
                nstates == emission.shape[0]
            if not pi is None:
                assert emission.shape[0] == pi.size

        if not pi is None:
            assert pi.ndim == 1
            if not nstates is None:
                assert nstates == pi.size
            else:
                nstates = pi.size

        if transition is None:
            transition = almost_uniform_matrix(nstates)

        if emission is None:
            emission = almost_uniform_matrix(nstates, nsymbols)

        if pi is None:
            pi = almost_uniform_vector(nstates)

        self.set_transition(transition)
        self.set_emission(emission)
        self.set_pi(pi)

    def set_transition(self, transition):
        assert is_stochastic(transition)
        self._transition = normalize(transition, dtype=self._REALTYPE)

    def set_emission(self, emission):
        assert is_stochastic(emission)
        self._emission = normalize(emission, dtype=self._REALTYPE)

    def set_pi(self, pi):
        assert is_stochastic(pi)
        self._pi = normalize(pi, dtype=self._REALTYPE)

    def get_transition(self):
        return self._transition

    def get_emission(self):
        return self._emission

    def get_pi(self):
        return self._pi

    def get_nstates(self):
        return self._pi.size

    def get_nsymbols(self):
        return self._emission.shape[1]

    def forward_backward(self, obseq):
        # local variables
        T = len(obseq)
        scalers = numpy.zeros(T);
        alphatilde = numpy.zeros((T, self.get_nstates()))
        alphahat = numpy.zeros((T, self.get_nstates()))
        betatilde = numpy.zeros((T, self.get_nstates()))
        betahat = numpy.zeros((T, self.get_nstates()))
        gammahat = numpy.zeros((T, self.get_nstates()))
        epsilonhat = numpy.zeros((T-1, self.get_nstates(), self.get_nstates()))
        
        # compute forward (alpha) variables
        for time in xrange(T):
            if time == 0:
                alphatilde[time,...] = self._pi*self._emission[...,obseq[time]]
            else:
                alphatilde[time,...] = numpy.sum(numpy.array([alphahat[time-1,...],]*self.get_nstates()).T*self._transition, axis=0)*self._emission[...,obseq[time]]
            scalers[time] = 1/alphatilde[time,...].sum()
            alphahat[time,...] = scalers[time]*alphatilde[time,...]
            
        # compute backward (beta) variables
        for time in xrange(T):
            reverse_time = T-1-time
            if reverse_time == T-1:
                betatilde[reverse_time,...] = 1
            else:
                betatilde[reverse_time,...] = numpy.sum(self._transition*numpy.array([self._emission[...,obseq[reverse_time+1]]*betahat[reverse_time+1,...],]*self.get_nstates()),axis=1)
            betahat[reverse_time,...] = scalers[reverse_time]*betatilde[reverse_time,...]

        # compute epsilon and gamma terms
        gammahat = alphahat*betahat/numpy.array([scalers,]*self.get_nstates()).T
        for time in xrange(T):
            if time < T-1:
                epsilonhat[time,...,...] = numpy.array([alphahat[time,...],]*self.get_nstates()).T*self._transition*numpy.array([self._emission[...,obseq[time+1]]*betahat[time+1,...],]*self.get_nstates())

        # compute likelihood
        likelihood = -numpy.sum(numpy.log(scalers))

        # render results
        return {"alphahat":alphahat, "betahat":betahat, "gammahat":gammahat, "epsilonhat":epsilonhat, "likelihood":likelihood}

    def do_baumwelch(self, obseqs):
        # likelihood of learned model
        likelihood = 0 

        # initial distribution of learned model
        pi = 0*self._pi

        # number of observations altogether
        L = 0

        # auxiliary tensors
        u = numpy.zeros(self._transition.shape) 
        w = numpy.zeros(self._emission.shape)
        v = numpy.zeros(self._pi.shape);
        x = numpy.zeros(self._pi.shape);

        # we'll now process all the observations and learn a new model
        for obseq in obseqs:
            L += 1

            # get length of observation sequence being processed
            T = len(obseq)

            # run Forwad-Backward algorithm
            fb = self.forward_backward(obseq)

            # update likelihood
            likelihood += fb.get('likelihood')

            # update pi
            pi += fb.get('gammahat')[0,...]

            # update auxiliary tensors
            u += fb.get('epsilonhat').sum(axis=0)
            v += fb.get('gammahat')[0:T-1,...].sum(axis=0)
            x += fb.get('gammahat').sum(axis=0)
            for j in xrange(self.get_nsymbols()):
                w[...,j] += fb.get('gammahat')[numpy.nonzero(numpy.array(obseq)==j)[0],...].sum(axis=0) 
                
        # compute learned model parameters
        pi /= L
        transition = u/numpy.array([v,]*u.shape[1]).T
        emission = w/numpy.array([x,]*w.shape[1]).T

        # render results
        return {'transition':transition, 'emission':emission, 'pi':pi, 'likelihood':likelihood}

    def learn(self,
              obseqs,
              tolerance=1e-7,
              miniterations=50,
              maxiterations=100,
              method='baumwelch'):
        iteration = 0
        likelihood = -numpy.inf

        while True:
            print "Iteration:", iteration
            print "Current model:", str(self)

            # budget exhausted ?
            if maxiterations <= iteration:
                print "Model did not converge after %d iterations."%iteration
                break

            # learn new model
            result = self.do_baumwelch(obseqs)

            print "Model likelihood:", result.get('likelihood')

            # converged ?
            if result.get('likelihood') == 0 and miniterations <= iteration:
                print "Model converged (to global optimum) after %d iterations."%iteration
                break

            # update model likelihood
            relative_gain = (result.get('likelihood') - likelihood)/numpy.abs(result.get('likelihood'))
            likelihood = result.get('likelihood')
            assert relative_gain >= 0 # if this fails, then somethx is terribly wrong with the code!!!

            print "Relative gain in model likelihood over last iteration:", relative_gain
            print 

            # converged ?
            if relative_gain < tolerance and miniterations <= iteration:
                print "Model converged after %d iterations (tolerance was set to %s)."%(iteration,tolerance)
                break

            # update model
            self.set_transition(result.get('transition'))
            self.set_emission(result.get('emission'))
            self.set_pi(result.get('pi'))

            # proceed with next iteration
            iteration += 1

    def viterbi_decode(self, obseq):
        T = len(obseq)
        hiddenseq = numpy.zeros(T);
        delta = numpy.zeros((T,self.get_nstates())) # likelihoods of all paths (incomplete) presently under consideration
        phi = numpy.zeros((T,self.get_nstates()))

        # take logs of probability terms so we don't suffer underflow, etc.
        logtransition = numpy.log(self._transition)
        logemission = numpy.log(self._emission)
        logpi = numpy.log(self._pi)

        # compute stuff for time = 0
        delta[0,...] = logpi + logemission[...,obseq[0]]
        phi[0,...] = numpy.arange(self.get_nstates())
        
        # run Viterbi decoder proper
        for time in xrange(1,T):
            for j in xrange(self.get_nstates()):
                tmp = delta[time-1,...] + logtransition[...,j]
                x = numpy.argmax(tmp)
                delta[time,j] = tmp[x] + logemission[j,obseq[time]]
                phi[time,j] = x

        # set last node of optimal path
        hiddenseq[T-1] = numpy.argmax(delta[T-1,...])
        likelihood = delta[T-1,hiddenseq[T-1]]
        
        # backtrack
        for time in xrange(T-1):
            reverse_time = T-2-time
            hiddenseq[reverse_time] = phi[reverse_time+1,hiddenseq[reverse_time+1]]

        # render results
        return {'states':list(hiddenseq), 'likelihood':likelihood}

    def __dict__(self):
        return {'transition':self._transition, 'emission':self._emission, 'pi':self._pi}

    def __str__(self):
        return '%s'%self.__dict__
    

class TestDiscreteHMM(unittest.TestCase):
    # def test_constructor_with_matrices(self):
    #     dhmm = DiscreteHMM(transition=numpy.mat([[0.6, 0.4],[0.3, 0.7]]), emission=numpy.mat([[1, 1],[2, 3]]), pi=numpy.array([5,6]))
    #     self.assertTrue(is_stochastic(dhmm.get_transition()))
    #     self.assertTrue(is_stochastic(dhmm.get_emission()))
    #     self.assertTrue(is_stochastic(dhmm.get_pi()))
    #     self.assertEqual(dhmm.get_nstates(), 2)
    #     self.assertEqual(dhmm.get_nsymbols(), 2)
        
    # def test_constructor_with_sizes(self):
    #     dhmm = DiscreteHMM(pi=numpy.array([0.6, 0.4]), nsymbols=3)
    #     self.assertTrue(is_stochastic(dhmm.get_transition()))
    #     self.assertTrue(is_stochastic(dhmm.get_emission()))
    #     self.assertTrue(is_stochastic(dhmm.get_pi()))
    #     self.assertEqual(dhmm.get_nstates(), 2)
    #     self.assertEqual(dhmm.get_nsymbols(), 3)

    def test_learn(self):
        dhmm = DiscreteHMM(transition=numpy.array([[0.5,0.5],[0.5,0.5]]),
                           emission=numpy.array([[1,1,1],[1,1,1]], dtype='float64'),
                           pi=numpy.array([0.6,0.4]),
                           )
          
        result = dhmm.do_baumwelch([[0,1,0,0,1,1]])
        print result
        dhmm.set_pi(result.get('pi'))
        dhmm.set_transition(result.get('transition'))
        dhmm.set_emission(result.get('emission'))

        result = dhmm.forward_backward([0,1,0,0,1,1])
        print result
        

    # # def test_viterbi(self):
    # #     dhmm = DiscreteHMM(transition=numpy.array([[0.4,0.6],[0.6,0.4]]),
    # #                        emission=numpy.array([[0.67,0.33],[0.41,0.59]]),
    # #                        pi=numpy.array([0.75,0.25]),
    # #                        )

    # #     self.assertEqual(dhmm.viterbi_decode([1,0,0,1,0,1]).get('states'), [0,1,0,1,0,1])
              
def main():
    dhmm = DiscreteHMM(nstates=2, nsymbols=26)
    lessons = list(chopper('data/corpus_words.dat'))
    numpy.random.shuffle(lessons)
    dhmm.learn(lessons[:500],maxiterations=100)
    
    print 
    print 'Viterbi classification of 26 symbols (cf. letters of the English alphabet):'
    clusters = dict()
    for i in xrange(26):
        letter = chr(ord('A')+i)
        class_label = dhmm.viterbi_decode([i]).get('states')[0]
        if i == 0:
            # 'a' is a vowel and 'a' is not a consonant
            clusters[class_label] = 'vowel'
            clusters[1-class_label] = 'consonant'
        print '\t%s is a %s'%(letter,clusters[class_label])

if __name__ == '__main__':
    if 'TEST' in os.environ:
        unittest.main()
    else:
        main()







