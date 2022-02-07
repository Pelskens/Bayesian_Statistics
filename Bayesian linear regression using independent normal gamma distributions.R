# Bayesian linear regression using independent normal gamma distributions 
# ELSKENS Philippe
# 01/06/2021

# Load packages
library(mvtnorm)

#-------------------------------------------------------------------------------
# Bayesian linear regression using independent normal gamma distributions
#-------------------------------------------------------------------------------
bayesLM <- function(Data, Prior, Mcmc) {
  set.seed(2021)
  # Extract the number of draws from posterior
  n_draws <- Mcmc$R / Mcmc$keep
  
  # Calculate prior nu and sigma squared bes on prior shape and rate.
  nu0 <- 2 * Prior$shape
  s_squared0 <- (2 * Prior$rate) / nu0
  h <- 1/2
  
  # Vectors to store the value of betas and h
  beta <- matrix(nrow = n_draws, ncol=length(Prior$mean))
  h_vector = rep(NA, n_draws)
  
  # Set a counter to save each the desire draws
  c <- 1 
  
  for (i in 1:Mcmc$R) { 
    # Calculate posterior V
    V1 <- solve(solve(Prior$varcov) + (h * t(Data$X) %*% Data$X))
    
    # Calculate the means of the posterior betas (Prior$mean is the prior betas)
    betas1<- V1 %*% (solve(Prior$varcov) %*% Prior$mean + 
                       h * t(Data$X) %*% Data$y)
    
    # Conditional of beta on y and h
    betas <- rmvnorm(1, mean = betas1, sigma = V1)
    
    # Calculate the posterior nu
    nu1 <- length(Data$y) + nu0
    
    # calculate the posterior s^2
    s_squared1 <- (t(Data$y - Data$X %*% t(betas)) %*% 
                     (Data$y - Data$X %*% t(betas)) + (nu0 * s_squared0)) / nu1
    
    # Conditional of h on y and beta
    shape1 = nu1/2
    rate1 = (s_squared1 * nu1) /2
    h <- rgamma(1, shape = shape1, rate = rate1)
    
    if (i %% Mcmc$keep == 0){
      beta[c,] <- betas
      h_vector[c] <- h
      c <- c + 1 
    }
  } 
  output = list("beta" = beta, "sigmasq" = 1/h_vector)
  return(output)
} 

#-------------------------------------------------------------------------------
# Simulate data points and call the function 
#-------------------------------------------------------------------------------
beta <- c(-1,2,3)
n_simulated <- 100
x <- cbind(1, matrix(runif(n=2*n_simulated, min=-8,max=8), nrow=100))
sigma_squared <- 2
y <- x%*%beta + rnorm(n=n_simulated, mean=0, sd=sqrt(sigma_squared))
k <- length(beta)

Data = list("y" = y, "X" = x)
Prior = list("mean" = c(0,0,0), "varcov"= 10*diag(k), "shape" = 0.5, "rate" = 2)
Mcmc = list("R" = 50000, "keep" = 1)

draws <- bayesLM(Data = Data, Prior = Prior, Mcmc = Mcmc)

#-------------------------------------------------------------------------------
# Investigate the output of function
#-------------------------------------------------------------------------------
par(mfrow = c(2, 2))
plot(seq(1,length(draws$sigmasq)), draws$beta[,1], type = "l", xlab = "Iteration",
     ylab = expression(paste("Value for ", beta[1])))
abline(h = mean(draws$beta[,1]), lty = 2, col = "red")
plot(seq(1,length(draws$sigmasq)), draws$beta[,2], type = "l", xlab = "Iteration",
     ylab = expression(paste("Value for ", beta[2])))
abline(h = mean(draws$beta[,2]), lty = 2, col = "red")
plot(seq(1,length(draws$sigmasq)), draws$beta[,3], type = "l", xlab = "Iteration",
     ylab = expression(paste("Value for ", beta[3])))
abline(h = mean(draws$beta[,3]), lty = 2, col = "red")
plot(seq(1,length(draws$sigmasq)), draws$sigmasq, type = "l", xlab = "Iteration",
     ylab = expression(paste("Value for ", sigma^2)))
abline(h = mean(draws$sigmasq), lty = 2, col = "red")
par(mfrow = c(1, 1))

#-------------------------------------------------------------------------------
# Calculate the Bayes estimate and credible intervals for all model parameters.
#-------------------------------------------------------------------------------
bayes_estimates_intervals <- cbind(mean(draws$beta[,1]), mean(draws$beta[,2]), 
                                   mean(draws$beta[,3]), mean(sqrt(draws$sigmasq)))
s_interval <- quantile(draws$sigmasq,probs=c(.025,.975))
b_interval <- apply(X=draws$beta,MARGIN=2,FUN=quantile,probs=c(.025,.975))
intervals <- cbind(b_interval,s_interval)
bayes_estimates_intervals <- rbind(bayes_estimates_intervals,intervals)
colnames(bayes_estimates_intervals ) <- c("beta_1", "beta_2", "beta_3", 
                                          "sigma") 
rownames(bayes_estimates_intervals ) <- c("Bayes estimates", "Lower bound", 
                                          "Upper bound")
bayes_estimates_intervals 

#-------------------------------------------------------------------------------
# Plot the marginal posterior distributions of all model parameters and show the 
# credible interval.
#-------------------------------------------------------------------------------
par(mfrow = c(2, 2))
plot(density(draws$beta[,1]), main = expression(paste("Density of ", beta[1])), 
     xlab = expression(paste("Value for ", beta[1])))
abline(v = b_interval[,1], lty = 2, col = "red")
plot(density(draws$beta[,2]), main = expression(paste("Density of ", beta[2])), 
     xlab = expression(paste("Value for ", beta[2])))
abline(v = b_interval[,2], lty = 2, col = "red")
plot(density(draws$beta[,3]), main = expression(paste("Density of ", beta[3])), 
     xlab = expression(paste("Value for ", beta[3])))
abline(v = b_interval[,3], lty = 2, col = "red")
plot(density(draws$sigmasq), main = expression(paste("Density of ",  sigma^2)),
     xlab = expression(paste("Value for ",  sigma^2)))
abline(v = c(s_interval), lty = 2, col = "red")
par(mfrow = c(1, 1))

#-------------------------------------------------------------------------------
# Make a plot of your dependent variable
#-------------------------------------------------------------------------------
z <- seq(-10,10,0.05)
lwr <- rep(NA,length(z))
upr <- rep(NA,length(z))
x2 <- runif(length(draws$sigmasq),-8,8)
pred <- matrix(nrow = length(draws$sigmasq), ncol = length(z))
for (i in(1:length(z))){
  pred[,i] <- draws$beta[,1] + draws$beta[,2]*z[i] + draws$beta[,3]*x2
  lwr[i] <- quantile(pred[,i],probs=c(.025,.975),na.rm=TRUE)[1]
  upr[i] <- quantile(pred[,i],probs=c(.025,.975),na.rm=TRUE)[2]
}

sigma <- sqrt(draws$sigmasq)
plot(x[,2],y,xlim = c(-10,10), ylim = c(-40,40), xlab = "X1", ylab = "y")
points(x[,2],y,col="red",pch=19,lwd=3)
lines(z, lwr, col="blue", lty=2)
lines(z, upr, col="blue", lty=2)
abline(h=0, v=0)
abline(a=mean(draws$beta[,1]),b=mean(draws$beta[,2]), col="orange", lwd=2)
legend(-10,35, legend=c("Regression line", "Prediction interval"),
       col=c("orange", "blue"), lty=1:2, cex=0.8)