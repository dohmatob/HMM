## +++++++++++++ ALGORITHMS FOR HMM (HIDDEN MARKOV MODEL) CALCULUS +++++++++++++
##                       (c) DOP (dohmatob elvis dopgima)

################################################
# PICKS VALUE FROM DISTRIBUTION ON FINITE SET
################################################
pick <- function(values, probabilities)
  {
    return (sample(values,1,replace=TRUE,prob=probabilities))
  }

#######################################################
# FUNCTION TO GENERATE OBSERVATION SEQUENCE FROM HMM
#######################################################
generate <- function(observables, transition, emission, initial,T)
  {
    n <- length(initial)
    o <- ncol(B)
    t <- 1
    start <- pick(1:n,initial)
    seq <- c()
    while (t<=T)
      {
        seq <- append(seq,pick(1:o,emission[start,]))
        start <- pick(1:n,transition[start,])
        t <- t+1
      }

    return (observables[seq])
  }

########################################################
# FUNCTION TO GENERATE OBSERVATION SEQUENCES FROM HMM
########################################################
generatebatch <- function(observables, transition, emission, initial,
                          N, # number of observation sequences to generate
                          l=2, # infimum length of observation sequence
                          L=20 # suprimum length of observation sequence
                          )
  {
    seqs <- list()
    count <- 0
    ldensity <- rep(1/(L-l+1),(L-l+1))
    while (count<N)
      {
        T <- pick(l:L,ldensity)
        seq <- generate(observables, transition, emission, initial, T)
        seqs[[count+1]] <- seq
        count <- count+1
      }

    return (seqs)
  }

##########################
# THE VITERBI ALGORITHM
##########################
viterbi <- function(observables, # states that can be observed
                    hidden, # hidden/internal states
                    observation, transition, emission,
                    initial=NULL # initial distribution 
                    )
{
  ## --[ some corrections ]--
  n <- length(hidden) # number of (hidden/internal) states
  T <- length(observation) 
  if (length(initial)==0 || T<2)
    {
      initial <- rep(1,n)/n
    }
  
  ## --[ take logs of all propabilities, so we don't suffer underflow, etc. ]--
  log2initial <- log2(initial)
  log2transition <- log2(transition)
  log2emission <- log2(emission)
  
  ## --[ initializations ]--
  delta <- matrix(0,ncol=n,nrow=T)
  phi <- matrix(rep(hidden,T),byrow=T,ncol=n)
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
  return (list(scalers=scalers,alphahat=alphahat,betahat=betahat,loglikelihood=loglikelihood,probability=probability,gammahat=gammahat,epsilonhat=epsilonhat))
}

learn <- function(observables, hidden, observations,transition, emission, initial)
  {
    ## --[ initialization ]--
    loglikelihood <- 0
    no <- length(observations)
    n <- length(hidden)
    m <- length(observables)
    A <- matrix(0,ncol=n,nrow=n) # transition matrix of next model
    B <- matrix(0,ncol=m,nrow=n) # emission matrix
    d <- array(0,dim=c(n))
    u <- array(0, dim=c(no,n,n))
    v <- array(0, dim=c(no,n))
    w <- array(0, dim=c(no,n,m))
    x <- array(0, dim=c(no,n))

    ## --[ process all observations ]--
    for (o in 1:no)
      {
        observation <- observations[[o]]
        T <- length(observation)
        fb <- forwardbackward(observables,hidden,observation,transition,emission,initial)
        loglikelihood <- loglikelihood+fb$loglikelihood
        for (i in 1:n)
          {
            d[i] <- d[i]+fb$gamma[1,i]
            for (t in 1:T)
              {
                x[o,i] <- x[o,i]+fb$gamma[t,i]
                if (t<T)
                  {
                    v[o,i] <- v[o,i]+fb$gamma[t,i]
                  }
              }
            for (j in 1:n)
                {
                  if (T>1)
                    {
                      for (t in 1:(T-1))
                        {
                          u[o,i,j] <- u[o,i,j]+fb$epsilon[t,i,j]
                        }
                    }
                }
            for (j in 1:m)
              {
                for (t in 1:T)
                  {
                    if (observables[j]==observation[t])
                      {
                        w[o,i,j] <- w[o,i,j]+fb$gamma[t,i]
                      }
                  }
              }
          }
      }

    ## --[ compute learned HMM parameters ]--
    pi <- d/no
    for (i in 1:n)
      {
        for (j in 1:n)
          {
            A[i,j] <- sum(u[,i,j])/sum(v[,i])
          }
        for (j in 1:m)
          {
            B[i,j] <- sum(w[,i,j])/sum(x[,i])
          }
      }

    ## --[ render results ]--
    return (list(loglikelihood=loglikelihood,transition=A,emission=B,initial=pi))
  }

###########################################################
# THE BAUM-WELCH EM (EXPECTATION MAXIMIZATION) ALGORITHM
###########################################################
baumwelch <- function(observables, hidden, observations, transition, emission, initial,
                      maxiter=1000, # computation budget
                      tol=1e-3 # tolerance level for convergence
                      )
{
  iter <- 0
  loglikelihood <- -Inf

  while (TRUE)
    {
      ## --[ check whether we still have computation budget left ]--
      if (iter>maxiter)
        {
          print ('OUT OF COMPUTATION BUDGET')
          break
        }
      print(list(iteration=iter,transition=transition,emission=emission,initial=initial))      
      iter <- iter+1

      ## --[ learn ]--
      model <- learn(observables, hidden, observations, transition, emission, initial)

      ## --[ housekeeping ]--
      if (model$loglikelihood==0)
        {
          print(list(loglikelihood=model$loglikelihood))
          print ('CONVERGED (TO GLOBAL OPTIMUM).')
          break
        }
      relativegain <- (model$loglikelihood-loglikelihood)/abs(model$loglikelihood)
      print(list(loglikelihood=model$loglikelihood,relativegain=relativegain))
      transition <- model$transition
      emission <- model$emission
      initial <- model$initial
      loglikelihood <- model$loglikelihood
      if (relativegain<tol)
        {
          print ('CONVERGED.')
          break
        }
    }
  
  return (list(transition=transition,emission=emission,initial=initial,loglikelihood=loglikelihood))
}

#################
# A DUMMY DEMO
#################
demo <- function()
{
  hidden <- 1:3 # hidden states
  observables <- 0:1 # observable states
  transition <- matrix(c(3,5,2,0,3,7,0,0,1)/10,byrow=T,ncol=3)
  emission <- matrix(c(2,0,1,1,0,2)/2,byrow=T,ncol=2)
  initial <- c(6,4,0)/10 # initial distribution
  observations <- list(c(0,0,1,1))
  
  ## --[ run baumwelch ]--
  solution <- baumwelch(observables,hidden,observations,transition,emission,initial)
}
