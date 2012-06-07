## +++++++++++++ ALGORITHMS FOR HMM (HIDDEN MARKOV MODEL) CALCULUS +++++++++++++
##                       (c) DOP (dohmatob elvis dopgima)

################################################
# PICKS VALUE FROM DISTRIBUTION ON FINITE SET
################################################
pick <- function(values, probabilities)
  {
    return (sample(values,1,replace=TRUE, prob=probabilities))
  }

#######################################################
# FUNCTION TO GENERATE OBSERVATION SEQUENCE FROM HMM
#######################################################
generate <- function(transition, emission, initial,T)
  {
    n <- length(initial)
    o <- ncol(B)
    t <- 1
    start <- pick(S,initial)
    seq <- c()
    while (t<=T)
      {
        seq <- append(seq,pick(1:o,emission[start,]))
        start <- pick(S,transition[start,])
        t <- t+1
      }

    return (seq)
  }

##########################
# THE VITERBI ALGORITHM
##########################
viterbi <- function(observables, # states that can be observed
                    hidden, # hidden/internal states
                    observation, transition, emission,
                    initial # initial distribution 
                    )
{
  ## --[ take logs of all propabilities, so we don't suffer underflow, etc. ]--
  log2initial <- log2(initial)
  log2transition <- log2(transition)
  log2emission <- log2(emission)
  
  ## --[ initializations ]--
  n <- length(hidden) # number of (hidden/internal) states
  T <- length(observation) 
  delta <- matrix(0,ncol=n,nrow=T)
  phi <- delta
  delta[1,] <- log2initial+log2emission[,which(observables==observation[1])]

  ## --[ iteratively construct most probable path ]--
  t <- 2 # really, t-1 is time
  while (t<=T)
    {
      j <- 1
      while (j<=n)
        {
          delta[t,j] <- max(delta[t-1,]+log2transition[,j])+log2emission[j,which(observables==observation[t])]
          phi[t,j] <- hidden[which.max(delta[t-1,]+log2transition[,j])]
          j <- j+1
        }
      t <- t+1
    }

  ## --[ calculate probability of MAP (Maximum A Posteri) path ]--
  probability <- 2**max(delta[T,])

  ## --[ gather-up results ]-- 
  state <- hidden[which.max(delta[T,])]
  path=c(state) # MAP path
  t <- T-1
  while (t>0)
    {
      state <- phi[t+1,which(hidden==state)]
      path <- c(state,path)
      t <- t-1
}
  
  return (list(path=path,probability=probability))
}

##############################################
# THE BAUM-WELCH FORWARD-BACKWARD ALGORITHM
##############################################
forwardbackward <- function(observables, hidden, observation, transition, emission, initial)
{
  ## --[ initializations ]--
  n <- length(initial)
  T <- length(observation)
  scalers <- rep(0,T) # these things will prevent underflow and ease the calculus
  epsilonhat <- array(0,dim=c(T-1,n,n))
  gammahat <- array(0,dim=c(T,n))
  alphatilde <- matrix(0,nrow=T,ncol=n)
  alphahat <- alphatilde
  betatilde <- alphatilde
  betahat <- betatilde

  ## --[ compute forward (alpha) parameters ]--
  for (t in 1:T)
    {
      if (t==1)
        {
            alphatilde[t,] <- initial*emission[,which(observables==observation[t])]
            scalers[t] <- 1/sum(alphatilde[t,])
            alphahat[t,] <- scalers[t]*alphatilde[t,]
          }
      else
        {
          for (i in 1:n)
            {
              alphatilde[t,i] <- sum(alphahat[t-1,]*transition[,i])*emission[i,which(observables==observation[t])]
            }
          scalers[t] <- 1/sum(alphatilde[t,])
          alphahat[t,] <- scalers[t]*alphatilde[t,]
        }
    }

  ## --[ compute backward (beta) parameters ]--
  for (t in T:1)
    {
      if (t==T)
        {
          betatilde[T,] <- rep(1,n)
          betahat[T,] <- scalers[T]*betatilde[T,]
        }
      else
        {
          for (i in 1:n)
            {
              betatilde[t,i] <- sum(transition[i,]*emission[,which(observables==observation[t+1])]*betahat[t+1,])
            }  
          betahat[t,] <- scalers[t]*betatilde[t,]
        }
    }

  ## --[ compute epsilon and gamma terms ]--
  for (t in  1:T)
    {
      gammahat[t,] <- alphahat[t,]*betahat[t,]/scalers[t]
      if (t<T)
        {
          for (i in 1:n)
            {
              for (j in 1:n)
                {
                  epsilonhat[t,i,j] <- alphahat[t,i]*transition[i,j]*emission[j,which(observables==observation[t+1])]*betahat[t+1,j]
                }
            }
        }
    }

  ## --[ compute log-likelihood (resp. probability) of observation sequence ]--
  loglikelihood <- -sum(log(scalers))
  probability <- exp(loglikelihood)

  ## --[ render results ]--
  return (list(alphahat=alphahat,betahat=betahat,loglikelihood=loglikelihood,probability=probability,gammahat=gammahat,epsilonhat=epsilonhat))
}
          
###########################################################
# THE BAUM-WELCH EM (EXPECTATION MAXIMIZATION) ALGORITHM
###########################################################
baumwelch <- function(observables, hidden, observations, transition, emission, initial,
                      iter=100, # computation budget
                      tol=1e-7 # tolerance level for convergence
                      )
{
  lo <- length(observables)
  n <- length(initial)
  L <- length(observations)
  loglikelihood <- -Inf
  count <- 0
  while (TRUE)
    {
      if (count>iter)
        {
          print("OUT OF COMPUTATION BUDGET!")
          break
        }

      print(list(iteration=count,transition=transition,emission=emission,initial=initial))
      A <- transition*0
      B <- emission*0
      d <- initial*0
      u <- array(0, dim=c(L,n,n))
      v <- array(0, dim=c(L,n))
      w <- array(0, dim=c(L,n,lo))
      x <- v
      for (l in 1:L)
        {
          observation <- observations[[l]]
          T <- length(observation)
          fb <- forwardbackward(observables, hidden, observation, transition, emission, initial)
          for (i in  1:n)
            {
              d[i] <- d[i]+fb$gammahat[1,i]
              v[l,i] <- v[l,i]+sum(fb$gammahat[1:(T-1),i])
              x[l,i] <- x[l,i]+sum(fb$gammahat[,i])
              for (j in 1:n)
                {
                  u[l,i,j] <- u[l,i,j]+sum(fb$epsilonhat[,i,j])
                }
              
              for (z in 1:lo)
                {
                  for (t in 1:T)
                    {
                      if (observables[z]==observation[t])
                        {
                          w[l,i,z] <- w[l,i,z]+fb$gammahat[t,i]
                        }
                    }
                }
            }
        }
      
      d <- d/sum(d)
      for (i in 1:n)
        {
          for (j in 1:n)
            {
              A[i,j] <- sum(u[,i,j])/sum(v[,i])
            }
          for (z in 1:lo)
            {
              B[i,z] <- sum(w[,i,z])/sum(x[,i])
            }
        }
      
      gain <- fb$loglikelihood-loglikelihood
      print(list(loglikelihood=fb$loglikelihood,gain=gain))
      if (gain<tol)
        {
          print('CONVERGED.')
          break
        }

      loglikelihood <- fb$loglikelihood
      count <- count+1
      initial <- d
      transition <- A
      emission <- B
    }
}

#################
# A DUMMY DEMO
#################
demo <- function()
{
  S <- 0:3 # hidden states
  O <- c('R','B','Y') # observable states
  Y <- c("R","Y","B","B","R","Y","R") # observation
  A <- matrix(c(0,1,0,0,0,0.2,0.5,0.3,0,0,0.4,0.6,1,0,0,0),ncol=4,byrow=T) # transition
  B <- matrix(c(3,2,5,7,2,1,9,0,1,2,8,0)/10,ncol=3,byrow=T) # emission
  initial <- c(1,0,0,0) # initial distribution
  
  ## --[ run the algos ]--
  print (viterbi(O,S,Y,A,B,initial))
  print (baumwelch(O,S,list(Y),A,B,initial))
}
