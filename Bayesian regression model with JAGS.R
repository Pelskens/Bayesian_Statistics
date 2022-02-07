# Bayesian regression model with JAGS
# ELSKENS Philippe
# 01/06/2021

# Load package
#install.packages("rjags")
library("rjags")

# Open csv file
data <- read.csv('sales.csv')

#-------------------------------------------------------------------------------
## Frequentist t-test
#-------------------------------------------------------------------------------
# Student's t-test: equal but unknown variances
n <- length(data$top)                                        #equal sample sizes
sp <- sqrt((var(data$top) + var(data$bottom))/2)             #sample std
t <- (mean(data$top) - mean(data$bottom)) / (sp * sqrt(2/n)) #test statistic
df <- 2*n - 2                                                #degrees of freedom
p_value <- 2*pt(t, df, lower.tail = FALSE)                   #p-value
t
p_value

# Compare with built-in student's t-test
test <- t.test(data$top, data$bottom, var.equal = TRUE)
test

#-------------------------------------------------------------------------------
## Bayesian t-test
#-------------------------------------------------------------------------------
# Preparation
X <- c(rep(0, 31), rep(1, 31))                               #dummy coding
y <- c(data$bottom, data$top)                                #sales
y_scaled <- (y-mean(y)) / sd(y)                              #standardize data
nobs <- n*2                                                  #n of observations                                               

# Frequentist regression
summary(lm(y~X))

# Bayesian regression model with JAGS
#------------------------------------------------------------------0. No burn-in
modelstring_noburn <- "
model{
  # Likelihood:
  for (i in 1:nobs){
    y[i] ~ dnorm(beta0 + beta1 * X[i], precision)            #Y~N(b0+b1*X,sigma)
  }
  
  # Priors:
  beta0 ~ dnorm(0, 0.01)                                     #intercept
  beta1 ~ dnorm(0, 0.01)                                     #slope             
  sigma ~ dunif(0, 100)                                      #standard deviation
  precision <- pow(sigma, -2)                                #precision
}
"

model_noburn <- jags.model(textConnection(modelstring_noburn),
                           data=list('X'=X, 'y'=y_scaled, 'nobs'=nobs),
                           n.chains=3,                                    
                           n.adapt=0)                                            

# Run MCMC
out_noburn <- coda.samples(model_noburn,
                           variable.names=c('beta0', 'beta1', 'sigma'),
                           n.iter=1000)

# Diagnostic checks
plot(out_noburn)                                                          
gelman.plot(out_noburn)                                                   
geweke.plot(out_noburn)                                                   
acfplot(out_noburn)                                                       

#-----------------------------------------1. Vague prior: beta1 ~ dnorm(0, 0.01)
modelstring_vague <- "
model{
  # Likelihood:
  for (i in 1:nobs){
    y[i] ~ dnorm(beta0 + beta1 * X[i], precision)            #Y~N(b0+b1*X,sigma)
  }
  
  # Priors:
  beta0 ~ dnorm(0, 0.01)                                     #intercept
  beta1 ~ dnorm(0, 0.01)                                     #slope             
  sigma ~ dunif(0, 100)                                      #standard deviation
  precision <- pow(sigma, -2)                                #precision
}
"

model_vague <- jags.model(textConnection(modelstring_vague),
                          data=list('X'=X, 'y'=y_scaled, 'nobs'=nobs),
                          n.chains=3,
                          n.adapt=100,                      #burn-in 
)

# Run MCMC
out_vague <- coda.samples(model_vague,
                          variable.names=c('beta0', 'beta1', 'sigma'),
                          n.iter=10000,
                          n.thin = 10)

# Diagnostic checks
plot(out_vague)                                                                 
gelman.plot(out_vague)                                                          
geweke.plot(out_vague)                                                          
acfplot(out_vague)                                                              

# Results
plot(out_vague)
summary(out_vague)
HPDI_vague <- HPDinterval(out_vague, prob=0.95)                                 
HPDI_vague                                                                      
lower_vague <- (HPDI_vague[[1]][2,1] + 
                  HPDI_vague[[2]][2,1] + 
                  HPDI_vague[[3]][2,1]) / 3
upper_vague <- (HPDI_vague[[1]][2,2] + 
                  HPDI_vague[[2]][2,2] + 
                  HPDI_vague[[3]][2,2]) / 3
lower_vague
upper_vague

#--------------------------------------2.Informative prior beta1 ~ dnorm(0, 100)
# Bayesian regression model with JAGS
modelstring_informative <- "
model{
  # Likelihood:
  for (i in 1:nobs){
    y[i] ~ dnorm(beta0 + beta1 * X[i], precision)            #Y~N(b0+b1*X,sigma)
  }
  
  # Priors:
  beta0 ~ dnorm(0, 0.01)                                     #intercept
  beta1 ~ dnorm(0, 100)                                      #slope
  sigma ~ dunif(0, 100)                                      #standard deviation
  precision <- pow(sigma, -2)                                #precision
}
"

model_informative <- jags.model(textConnection(modelstring_informative),
                                data=list('X'=X, 'y'=y_scaled, 'nobs'=nobs),
                                n.chains=3, 
                                n.adapt=100)

# Run MCMC chain
out_informative <- coda.samples(model_informative,
                                variable.names=c('beta0', 'beta1', 'sigma'),
                                n.iter=10000,
                                n.thin = 10)

# Diagnostic checks
plot(out_informative) 
gelman.plot(out_informative) 
geweke.plot(out_informative) 
acfplot(out_informative) 

# Results
plot(out_informative)
summary(out_informative)
HPDI_informative <- HPDinterval(out_informative, prob=0.95)
HPDI_informative
lower_informative <- (HPDI_informative[[1]][2,1] + 
                        HPDI_informative[[2]][2,1] + 
                        HPDI_informative[[3]][2,1]) / 3
upper_informative <- (HPDI_informative[[1]][2,2] + 
                        HPDI_informative[[2]][2,2] + 
                        HPDI_informative[[3]][2,2]) / 3
lower_informative
upper_informative

#-----------------------------------------3. Boundary prior: beta1 ~ dnorm(0, 5)
modelstring_boundary <- "
model{
  # Likelihood:
  for (i in 1:nobs){
    y[i] ~ dnorm(beta0 + beta1 * X[i], precision)            #Y~N(b0+b1*X,sigma)
  }
  
  # Priors:
  beta0 ~ dnorm(0, 0.01)                                     #intercept
  beta1 ~ dnorm(0, 5)                                        #slope
  sigma ~ dunif(0, 100)                                      #standard deviation
  precision <- pow(sigma, -2)                                #precision
}
"

model_boundary <- jags.model(textConnection(modelstring_boundary),
                             data=list('X'=X, 'y'=y_scaled, 'nobs'=nobs),
                             n.chains=3,
                             n.adapt=100)

# Run MCMC chain
out_boundary <- coda.samples(model_boundary,
                             variable.names=c('beta0', 'beta1', 'sigma'),
                             n.iter=10000,
                             n.thin = 10)

# Diagnostic checks
plot(out_boundary)
gelman.plot(out_boundary)
geweke.plot(out_boundary)
acfplot(out_boundary)

# Results
plot(out_boundary)
summary(out_boundary)
HPDI_boundary <- HPDinterval(out_boundary, prob=0.95)
HPDI_boundary
lower_boundary <- (HPDI_boundary[[1]][2,1] + 
                     HPDI_boundary[[2]][2,1] + 
                     HPDI_boundary[[3]][2,1]) / 3
upper_boundary <- (HPDI_boundary[[1]][2,2] + 
                     HPDI_boundary[[2]][2,2] + 
                     HPDI_boundary[[3]][2,2]) / 3
lower_boundary
upper_boundary

#-------------------------------------------------------------------------------
## Bayes factor
#-------------------------------------------------------------------------------
# Preparation
k <- 1                                                       #number of coef

# Linear regression function with conjugate priors
blm <- function(X,y,nu0,V0,ssq0,beta0){
  # Calculate data quantities
  betahat <- chol2inv(chol(t(X)%*%X))%*%t(X)%*%y
  nu <- nobs-k
  ssq <- t(y-X%*%betahat)%*%(y-X%*%betahat)/nu
  
  # Posterior values
  nu1 <- nu0 + nobs
  V1 <- chol2inv(chol(t(X)%*%X+chol2inv(chol(V0))))
  betatilde <- V1 %*% (t(X)%*%X%*%betahat + chol2inv(chol(V0))%*%beta0)
  ssq1 <- (nu0*ssq0 + n*ssq + 
             t(betahat-beta0)%*%chol2inv(chol(V0+chol2inv(chol(t(X)%*%X))))
           %*%(betahat-beta0))/nu1 
  
  return(c(nu1, V1, ssq1))
}

# Function Bayes factor
Bayes_factor <- function(V0, V0_a, V1, V1_a, a1, a1_a, d1){
  (sqrt(det(V0))*sqrt(det(V1_a))) / 
    (sqrt(det(V0_a))*sqrt(det(V1))) * (a1/a1_a)**(d1/2)
}

#-------------------------------------------------------------------1. First run
# Prior values: Informative 'H0'
nu0 <- 100                                                 
ssq0 <- 5
beta0 <- rep(0,k)
V0 <- diag(k)*0.01                                         

# Prior values: Uninformative 'Ha'
nu0_a <- 0.01                                                                                                       
V0_a <- diag(k)*100                                        

# Run the model
post <- blm(y=y,X=X,nu0=nu0,ssq0=ssq0,beta0=beta0,V0=V0)                        
post_a <- blm(y=y,X=X,nu0=nu0_a,ssq0=ssq0,beta0=beta0,V0=V0_a)

# Unpack posterior values
nu1 <- post[1]
V1 <- as.matrix(post[2])
ssq1 <- post[3]

nu1_a <- post_a[1]
V1_a <- as.matrix(post_a[2])
ssq1_a <- post_a[3]

a1 <- nu1/2                                                  #shape
a1_a <- nu1_a/2
d1 <- 2/(nu1*ssq1)                                           #scale

# Calculate Bayes factor
B1 <- Bayes_factor(V0=V0, V0_a=V0_a, V1=V1, V1_a=V1_a, a1=a1, a1_a=a1_a, d1=d1)
B1

#------------------------------------------------------------------2. Second run
# Prior values that reflect 'H0' and 'Ha'
nu0 <- 10
ssq0 <- 5
beta0 <- rep(0,k)
V0 <- diag(k)*10

nu0_a <- 1                                                                    
V0_a <- diag(k)*35 

# Run the model
post <- blm(y=y,X=X,nu0=nu0,ssq0=ssq0,beta0=beta0,V0=V0)                        
post_a <- blm(y=y,X=X,nu0=nu0_a,ssq0=ssq0,beta0=beta0,V0=V0_a)

# Unpack posterior values
nu1 <- post[1]
V1 <- as.matrix(post[2])
ssq1 <- post[3]

nu1_a <- post_a[1]
V1_a <- as.matrix(post_a[2])
ssq1_a <- post_a[3]

a1 <- nu1/2                                                  #shape
a1_a <- nu1_a/2
d1 <- 2/(nu1*ssq1)                                           #scale

# Calculate Bayes factor
B2 <- Bayes_factor(V0=V0, V0_a=V0_a, V1=V1, V1_a=V1_a, a1=a1, a1_a=a1_a, d1=d1)
B2