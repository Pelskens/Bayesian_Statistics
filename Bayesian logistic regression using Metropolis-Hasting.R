# Bayesian logistic regression using Metropolis-Hasting
# ELSKENS Philippe
# 01/06/2021

# Load packages
#install.packages("bayesQR")
#install.packages("mvtnorm")
library("bayesQR")
library("mvtnorm")

#-------------------------------------------------------------------------------
# Bayesian logistic regression using Metropolis-Hasting
#-------------------------------------------------------------------------------
bayesLogReg <- function(Data, Prior, Mcmc) {
  set.seed(2021)
  
  # Initialize empty matrix
  n_draws <- Mcmc$R / Mcmc$keep
  beta_draws <-  matrix(ncol = length(Prior$mean), nrow = n_draws)
  
  # Calculate the Pi vector based on sigmoid function
  Pi = 1 / (1 + exp(-(as.matrix(Data$X) %*% Prior$mean)))
  
  # Calculate the log likelihood by converting the multiplication to sum
  loglikelihood <- sum(Pi*Data$y+(1-Pi)*(1-Data$y))
  
  # Calculate the initial prior density
  pri_density <- dmvnorm(Prior$mean, mean = Prior$mean, sigma = Prior$varcov,
                         log = TRUE)
  # Calculate the log of posterior density.
  pos_density <- exp(loglikelihood + pri_density)
  current <- pos_density
  
  # Set the prior mean as the starting point in the draws
  beta_draws[1,] <- Prior$mean
  
  n_rejection = 0
  c <- 1
  
  for (i in 2:Mcmc$R){
    # Draw from proposal distribution
    p_beta <- rmvnorm(1, Prior$mean, sigma = Mcmc$step * Prior$varcov)[1,]
    
    # Loglikelihood
    p_Pi <- 1 / (1 + exp(-(Data$X %*% p_beta)))
    
    p_loglike <- sum(p_Pi*Data$y + (1-p_Pi)*(1-Data$y))
    p_pos_density <- exp(p_loglike + pri_density)
    
    # Calculate probability of moving
    p <- min(p_pos_density / current, 1)
    
    # Accept or reject proposed position
    if (p == 1){
      current <- p_pos_density
      if (i %% Mcmc$keep == 0){
        beta_draws[c,] <- p_beta
        c <- c + 1
      }
    } else if (runif(n = 1) <= p) {
      current <- p_pos_density
      if (i %% Mcmc$keep == 0){
        beta_draws[c,] <- p_beta
        c <- c + 1
      }
    } else{
      if (i %% Mcmc$keep == 0){
        n_rejection <- n_rejection + 1
      }
    }
  }
  
  # Calculate and print rejection rate
  rejection_rate <- n_rejection / n_draws
  print('Rejection rate is:')
  print(rejection_rate)
  
  # Calculate the number of accepted samples and return the vector beta_draws
  n_accepted <- n_draws - n_rejection
  return(beta_draws[1:n_accepted, ])
}
#-------------------------------------------------------------------------------
# Prepare Input data and call the function 
#-------------------------------------------------------------------------------
# Assign data variables
data(Churn)
y <- Churn$churn
X <- cbind(1, Churn$gender, Churn$Social_Class_Score, Churn$lor, Churn$recency)
names <- c("intercept","gender","SCS","lor","recency")
k <- ncol(X)
Data <- list("y"=y, "X"=X)

# Set MCMC parameters
R <- 100000
keep <- 1
step <- 0.0008
Mcmc <- list("R" = R, "step" = step, "keep" = keep)

# Prior distribution
Prior <- list("mean" = c(0, 0, 0, 0, 0), "varcov" = 5 * diag(5))

# Call the function and store beta draws
beta <- bayesLogReg(Data,Prior,Mcmc)

# Trace plot 
indices <- seq(1:length(beta[,1]))
par(mfrow = c(3, 2))
for(i in 1:5){
  plot(indices, beta[,i], type = "l", xlab = "Iteration", ylab = names[i])
  abline(h = mean(beta[,i], na.rm=TRUE), lty = 2, col = "red")
}
par(mfrow = c(1, 1))

#-------------------------------------------------------------------------------
# Estimate a logistic regression model with churn as dependent variable.
#-------------------------------------------------------------------------------
estimates <- apply(beta,mean, MARGIN=2)
interval <- apply(X=beta, MARGIN=2, FUN=quantile,probs=c(.025,.975), na.rm=TRUE)

bayes_estimate <- rbind(estimates,interval)
colnames(bayes_estimate) <- names
rownames(bayes_estimate) <- c("mean","Lower bound", "Upper bound")
bayes_estimate

par(mfrow = c(3, 2))
for(i in 1:5){
  hist(beta[,i], breaks = 100, main = names[i])
  abline(v = bayes_estimate[2,i], lty = 2, col = "red")
  abline(v = bayes_estimate[3,i], lty = 2, col = "red")
}
par(mfrow = c(1, 1))

#-------------------------------------------------------------------------------
# What is the probability that beta_recency < 0 ?
#-------------------------------------------------------------------------------
probability <- sum(beta[,5] < 0, na.rm = TRUE)/length(beta[,5])
probability

#-------------------------------------------------------------------------------
# What is the probability that beta_gender > beta_lor ?
#-------------------------------------------------------------------------------
wilcox_test <- wilcox.test(beta[,2], beta[,4], paired = TRUE, alternative="greater") 
t_test <- t.test(beta[,2], beta[,4], paired = TRUE, alternative = "greater")
wilcox_test
t_test

#-------------------------------------------------------------------------------
# Gibbs sampler
#-------------------------------------------------------------------------------
# Calculate correlation between features                                               
cor(subset(Churn, select=(-churn)))

## Define functions
# Draws from the posterior distribution of the Bayesian probit model
# with normal prior on the regression coefficients.
# Taken from: https://github.com/driesbenoit/benchmark_R_Julia_Fortran
binprobbayes <- function(y,X,b0,B0,R){
  
  n <- nrow(X) 
  k <- ncol(X)
  
  B <- chol2inv(chol(chol2inv(chol(B0)) + t(X)%*%X))
  Bb <- chol2inv(chol(B0))%*%b0
  
  # Initialize space to save draws
  betadraw <- matrix(NA,nrow=R,ncol=k)
  
  # Set starting values
  beta <- rep(0, k)
  ystar <- rep(0,n)
  
  # Start mcmc
  for (i in 1:R){
    
    # Draw next value for ystar
    ystar = mapply(FUN=rtnorm,mu=X%*%beta,sd=1,positive=y)
    
    # Draw new value for beta
    beta <- B%*%(Bb + (t(X)%*%ystar)) + t(chol(B))%*%rnorm(n=k)
    
    # Save draw
    betadraw[i,] = beta
  }
  
  return(betadraw)
}

# Returns one draw from the truncated normal distribution
# Taken from: https://github.com/driesbenoit/benchmark_R_Julia_Fortran
rtnorm <- function(mu, sd, positive){
  if (positive){
    min = 0
    max = Inf
  }else{
    min = -Inf 
    max = 0
  }
  
  if(is.finite(max)){
    lt <- FALSE
    c <- -(max-mu)/sd
  }else{
    lt=TRUE
    c <- (min-mu)/sd
  }	
  
  if(c <= .45){
    # normal rejection sampling
    repeat{
      x <- rnorm(n=1, mean=0, sd=1)
      if(x>c)break
    }
  } else if (c > .45){
    # exponential rejection sampling
    repeat{
      x <- rexp(n=1, rate=c)
      u <- runif(n=1, min=0, max=1)
      if(u < exp(-.5*(x^2)))break
    }
    x <- x + c
  }
  
  if(lt){
    return(mu+sd*x)
  }else{
    return(mu-sd*x)
  }
}

# Execute algorithm
out_Gibbs_1 <- binprobbayes(y=y,X=X,c(0,0,0,0,0),diag(5)*5,10000)
out_Gibbs_2 <- binprobbayes(y=y,X=X,c(0,0,0,0,0),diag(5)*0.001,10000)

# Check Bayes estimate
colMeans(out_Gibbs_1)
colMeans(out_Gibbs_2)

# Plot trace plots
matplot(out_Gibbs_1,typ="l",lty=1)
matplot(out_Gibbs_2,typ="l",lty=1)
