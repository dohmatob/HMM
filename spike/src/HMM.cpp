#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/multi_array.hpp>
#include <assert.h>
#include <math.h>

using namespace boost::numeric::ublas;

typedef vector<int> intseq;

typedef struct argmax_struct
{
  double value;
  int index;
} argmax_t;

typedef struct path_struct
{
  intseq states;
  double likelihood;
} path_t;

vector<double> vlog2(const vector<double> &u)
{
  vector<double> v(u.size());
  
  for (int i = 0; i < u.size(); i++)
    {
      v(i) = log2(u(i));
    }

  return v;
}

matrix<double> mlog2(const matrix<double> &A)
{
  matrix<double> B(A.size1(),A.size2());
  
  for (int i = 0; i < A.size1(); i++)
    {
      row(B,i) = vlog2(row(A,i));
    }

  return B;
}

argmax_t argmax(vector<double> &u)
{
  argmax_t am;
  am.index = 0;
  am.value = u(0);

  for (int i = 1; i < u.size(); i++)
    {
      if (u(i) > am.value)
	{
	  am.value = u(i);
	  am.index = i;
	}
    }

  return am;
}
  
class HMM
{
private:
  int _nhidden;
  int _nobservables;
  matrix<double> _transition;
  matrix<double> _emission;
  vector<double> _pi;

  // logarithms, so we don't suffer 'underflow' in viterbi, etc.
  matrix<double> _log2transition;
  matrix<double> _log2emission;
  vector<double> _log2pi;

public:
  HMM(matrix<double> transition, matrix<double> emission, vector<double> pi)
  {
    // sanity checks
    // XXX add checks for stochasticity
    assert(pi.size()==emission.size1());
    assert(transition.size1()==emission.size1());
    assert(transition.size2()==emission.size1());

    _transition = transition;
    _emission = emission;
    _pi = pi;
    _log2transition = mlog2(transition);
    _log2emission = mlog2(emission);
    _log2pi = vlog2(_pi);
    _nhidden = pi.size();
    _nobservables = emission.size2();
  }

  const matrix<double> &getTransition(void)
  {
    return _transition;
  }

  const matrix<double> &getEmission(void)
  {
    return _emission;
  }

  const vector<double> &getPi(void)
  {
    return _pi;
  }

  int getNhidden(void)
  {
    return _nhidden;
  }

  int getNObservables(void)
  {
    return _nobservables;
  }

  void display(void)
  {
    std::cout << "transition = " << _transition << "\n\n";
    std::cout << "emission = " << _emission << "\n\n";
    std::cout << "pi = " << _pi << "\n\n";
  }

  bool isState(int i)
  {
    return 0 <= i && i < _nhidden;
  }

  /////////////////////////////////////////////////////
  // THE VITERBI MAP (MAXIMUM A POSTERIORI) DECODING
  /////////////////////////////////////////////////////
  path_t viterbi(const intseq &obseq)
  {
    // initializations
    int T = obseq.size(); 
    vector<int> hiddenseq(T); // optimal path (sequence of hidden states that generated observed trace)
    path_t path;
    int state;
    double proba;
    matrix<double> delta(T,_nhidden);
    matrix<int> phi(T,_nhidden);
    argmax_t am;
    vector<double> tmp(_nhidden);

    // compute stuff for time = 0
    assert(isState(obseq[0]));
    row(delta,0) = _log2pi+column(_log2emission,obseq[0]);
    for (int j = 0; j < _nhidden; j++)
      {
	phi(0,j) = j;
      }

    // run viterbi proper
    for (int time = 1; time < T; time++)
      {
	for (int j = 0; j < _nhidden; j++)
	  {
	    tmp = row(delta,time-1)+column(_log2transition,j);
	    am = argmax(tmp);
	    assert(isState(obseq[time]));
	    delta(time,j) = am.value+_log2emission(j,obseq[time]);
	    phi(time,j) = am.index;
	  }
      }

    // set last node on optima path
    tmp = row(delta,T-1);
    am = argmax(tmp);
    path.likelihood = am.value;
    state = am.index;
    hiddenseq(T-1) = state;

    // backtrack
    for (int time = T-2; time >= 0; time--)
      {
	state = phi(time+1,state);
	hiddenseq(time) = state;
      }
      
    path.states = hiddenseq;

    return path;
  }
};

int main(void)
{
  // XXX refactor main into unittest cases
  matrix<double> trans(7,7);
  matrix<double> em(7,7);
  vector<double> pi(7);
  vector<int> o(4);
  o(3) = 0;
  o(1) = o(2) = 1;
  o(0) = 3;

  for (int i = 0; i < trans.size1(); i++)
    {
      pi(i) = 1.0/7;
      
      for (int j = 0; j < trans.size2(); j++)
	{
	  em(i,j) = 1.0/7;
	  trans(i,j) = 1.0/7;
	}
    }

  trans(6,0) = 1;
  trans(6,1) = trans(6,2) = trans(6,3) = trans(6,4) = trans(6,5) = trans(6,6) = 0;
  vector<double> p = prod(pi,trans);
  
  // initialize HMM object
  HMM hmm(trans,em,pi);
  std::cout << "HMM paramters: " << std::endl;
  hmm.display();

  // run viterbi
  path_t path = hmm.viterbi(o);
  std::cout << "The most a posteriori most probable sequence of states that emitted the trace: " << o << " is " << path.states << ".\nIts likelihood is " << path.likelihood << std::endl;

  
  return 0;
}
