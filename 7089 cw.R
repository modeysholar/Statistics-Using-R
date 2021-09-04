library(ggplot2)
library(fBasics)

### Importing the dataset
library(readr)
x <- read_csv("x.csv",  col_names = c("x1", "x2", "x3", "x4"))

library(readr)
y <- read_csv("y.csv", col_names = "y")

library(readr)
time <- read_csv("time.csv", col_names = "Time")

EEG_dataset<- cbind(x, time, y)

#-------------------------------------------------------
### TASK 1; Performing an Exploratory Data Analysis
#-------------------------------------------------------
  
##----Investigating the Time series plots (of input and output EEG signals)----

ggplot(EEG_dataset, aes(Time, x1)) + geom_line() + labs(title = "Time Series Graph vs Input x1")  #plots the time series graph for input x1
ggplot(EEG_dataset, aes(Time, x2)) + geom_line() + labs(title = "Time Series Graph vs Input  x2")  #plots the time series graph for input x2
ggplot(EEG_dataset, aes(Time, x3)) + geom_line() + labs(title = "Time Series Graph vs Input x3")  #plots the time series graph for input x3
ggplot(EEG_dataset, aes(Time, x4)) + geom_line() + labs(title = "Time Series Graph vs Input x4")  #plots the time series graph for input x4
ggplot(EEG_dataset, aes(Time, y)) + geom_line() + labs(title = "Time Series Graph vs Output y")  #plots the time series graph for output y

##----Distribution for each EEG signal----

ggplot(EEG_dataset, aes(x1)) + geom_density() + labs(title = "Distribution Graph of input x1")  #plots the distribution graph for input x1
summary(EEG_dataset$x1)
skewness(EEG_dataset$x1)

ggplot(EEG_dataset, aes(x2)) + geom_density() + labs(title = "Distribution Graph of input x2") #plots the distribution graph for input x2
summary(EEG_dataset$x2)
skewness(EEG_dataset$x2)

ggplot(EEG_dataset, aes(x3)) + geom_density() + labs(title = "Distribution Graph of input x3")  #plots the distribution graph for input x3
summary(EEG_dataset$x3)
skewness(EEG_dataset$x3)

ggplot(EEG_dataset, aes(x4)) + geom_density() + labs(title = "Distribution Graph of input x4")  #plots the distribution graph for input x4
summary(EEG_dataset$x4)
skewness(EEG_dataset$x4)

ggplot(EEG_dataset, aes(y)) + geom_density() + labs(title = "Distribution Graph of input y")  #plots the distribution graph for output y
summary(EEG_dataset$y)
skewness(EEG_dataset$y)


##----Correlation and scatter plots (between different input EEG signals and the output EEG)----

ggplot(EEG_dataset, aes(x1, y)) + geom_point() + labs(title = "Scatter plot graph of Input x1 Against output y")  #plots the scatter plot for input x1
cor(EEG_dataset$x1, EEG_dataset$y) # computes the correlation value between input signal x1 & y

ggplot(EEG_dataset, aes(x2, y)) + geom_point() + labs(title = "Scatter plot graph of Input x2 Against output y")  #plots the scatter plot for input x2
cor(EEG_dataset$x2, EEG_dataset$y)

ggplot(EEG_dataset, aes(x3, y)) + geom_point() + labs(title = "Scatter plot graph of Input x3 Against output y")  #plots the scatter plot for input x3
cor(EEG_dataset$x3, EEG_dataset$y)

ggplot(EEG_dataset, aes(x4, y)) + geom_point() + labs(title = "Scatter plot graph of Input x4 Against output y")  #plots the scatter plot for input x4
cor(EEG_dataset$x4, EEG_dataset$y)



#------------------------------------------------------------------------
### TASK 2: REGRESSION- Modelling the relationship between EEG signals.
#------------------------------------------------------------------------
  
##----Task 2.1: Estimating model parameters----

# MODEL 1 : y= t_1*x4 + t_2*x1^2 + t_3*x1^3 + t_4*x3^4 + tbias + £
one = matrix(1 , nrow(x), 1)  
x1_sqr= matrix(EEG_dataset$x1^2)  # x1 ^ 2
x1_cube= matrix(EEG_dataset$x1^3)   # x1 ^ 3
x3_qud= matrix(EEG_dataset$x3^4)    # x3 ^ 4
x4= matrix(EEG_dataset$x4)
m1= cbind(x4, x1_sqr, x1_cube, x3_qud, one)  #combining variables of the polynomial
y= matrix(EEG_dataset$y)
thetaHat_1 = solve(t(m1) %*% m1) %*% t(m1) %*% y  #estimates Model1 parameters
print(thetaHat_1)


# MODEL 2 : y= t_1*x3^3 + t_2*x3^4 + tbias + £
x3_cube= matrix(EEG_dataset$x3^3)   # x3 ^ 3
m2= cbind(x3_cube, x3_qud, one)  #combining variables of the polynomial
thetaHat_2 = solve(t(m2) %*% m2) %*% t(m2) %*% y  #estimates Model2 parameters
print(thetaHat_2)


# MODEL 3: y= t_1*x2 + t_2*x1^3 + t_3*x3^4 + tbias + £
x2= matrix(EEG_dataset$x2)
m3= cbind(x2, x1_cube, x3_qud, one)  #combining variables of the polynomial
thetaHat_3 = solve(t(m3) %*% m3) %*% t(m3) %*% y  #estimates Model3 parameters
print(thetaHat_3)


# MODEL 4: y= t_1*x4 + t_2*x1^3 + t_4*x3^4 + tbias + £
m4= cbind(x4, x1_cube, x3_qud, one)  #combining variables of the polynomial
thetaHat_4 = solve(t(m4) %*% m4) %*% t(m4) %*% y  #estimates Model4 parameters
print(thetaHat_4)


# MODEL 5: y= t_1*x4 + t_2*x1^2 + t_3*x1^3 + t_4*x3^4 + t_5*x1^4  tbias + £
x1_qud= matrix(EEG_dataset$x1^4)   # x1 ^ 4
m5= cbind(x4,x1_sqr, x1_cube, x3_qud, x1_qud, one)  #combining variables of the polynomial
thetaHat_5 = solve(t(m5) %*% m5) %*% t(m5) %*% y  #estimates Model5 parameters
print(thetaHat_5)



##----Task 2.2: model residual (error) sum of squared errors(RSS)----

  #MODEL 1
RSS1 = sum((y-(m1 %*% thetaHat_1))^2)  #computes the RSS for candidate Model1
print(RSS1)

#MODEL 2
RSS2 = sum((y-(m2 %*% thetaHat_2))^2)  #computes the RSS for candidate Model2
print(RSS2)

#MODEL 3
RSS3 = sum((y-(m3 %*% thetaHat_3))^2)  #computes the RSS for candidate Model3
print(RSS3)

#MODEL 4
RSS4 = sum((y-(m4 %*% thetaHat_4))^2)  #computes the RSS for candidate Model4
print(RSS4)

#MODEL 5
RSS5 = sum((y-(m5 %*% thetaHat_5))^2)  #computes the RSS for candidate Model5
print(RSS5)



##----Task 2.3: Compute the log-likelihood function for every candidate model----
n<- 201

#Model 1
Var1= RSS1/(n-1)  #variance of model’s residuals
log_like1= (-(n/2)*log(2*pi))- ((n/2)*log(Var1)) - (((2*Var1)^-1)*RSS1) #computes log-likelihood function for Model1
print(log_like1)

#Model 2
Var2= RSS2/(n-1)  #variance of model’s residuals
log_like2= (-(n/2)*log(2*pi))- ((n/2)*log(Var2)) - (((2*Var2)^-1)*RSS2) #computes log-likelihood function for Model2
print(log_like2)

#Model 3
Var3= RSS3/(n-1)  #variance of model’s residuals
log_like3= (-(n/2)*log(2*pi))- ((n/2)*log(Var3)) - (((2*Var3)^-1)*RSS3) #computes log-likelihood function for Model3
print(log_like3)

#Model 4
Var4= RSS4/(n-1)  #variance of model’s residuals
log_like4= (-(n/2)*log(2*pi))- ((n/2)*log(Var4)) - (((2*Var4)^-1)*RSS4) #computes log-likelihood function for Model4
print(log_like1)

#Model 5
Var5= RSS5/(n-1)  #variance of model’s residuals
log_like5= (-(n/2)*log(2*pi))- ((n/2)*log(Var5)) - (((2*Var5)^-1)*RSS5) #computes log-likelihood function for Model5
print(log_like5)


##----Task 2.4: Compute the (AIC) and (BIC) for every candidate model:----

# Model1
k1= length(thetaHat_1)  #k1 is the nos of estimated parameters of model 1
AIC1= (2*k1)-(2*log_like1)  #Computes Models 1 AIC value
BIC1= (k1*log(n))-(2*log_like1) #Computes Model1 BIC Value
AIC1
BIC1

# Model2
k2= length(thetaHat_2)  #k2 is the nos of estimated parameters of model 2
AIC2= (2*k2)-(2*log_like2)  #Computes Model 2 AIC value
BIC2= (k2*log(n))-(2*log_like2) #Computes Model2 BIC Value
AIC2
BIC2

# Model3
k3= length(thetaHat_3)  #k3 is the nos of estimated parameters of model 3
AIC3= (2*k3)-(2*log_like3)  #Computes Model 3 AIC val
BIC3= (k3*log(n))-(2*log_like3) #Computes Model3 BIC Value
AIC3
BIC3

# Model4
k4= length(thetaHat_4)  #k4 is the nos of estimated parameters of model 4
AIC4= (2*k4)-(2*log_like4)  #Computes Model 4 AIC val
BIC4= (k4*log(n))-(2*log_like4) #Computes Model4 BIC Value
AIC4
BIC4

# Model5
k5= length(thetaHat_5)  #k5 is the nos of estimated parameters of model 5
AIC5= (2*k5)-(2*log_like5)  #Computes Model 5 AIC val
BIC5= (k5*log(n))-(2*log_like5) #Computes Model5 BIC Value
AIC5
BIC5


##----Task 2.5:Check the distribution of model prediction errors (residuals) for each candidate model.----

# Model1
y_Hat1 = m1 %*% thetaHat_1
error1 = matrix(y - y_Hat1) #Calculates residual error
qqnorm(error1)            #plots qq graph
qqline(error1, col="red")
kurtosis(error1)      #calculates the kurtosis value

# Model2
y_Hat2 = m2 %*% thetaHat_2
error2 = matrix(y - y_Hat2)
qqnorm(error2)
qqline(error2, col="red")
kurtosis(error2)

# Model3
y_Hat3 = m3 %*% thetaHat_3
error3 = matrix(y - y_Hat3)
qqnorm(error3)
qqline(error3, col="red")
kurtosis(error3)

# Model4
y_Hat4 = m4 %*% thetaHat_4
error4 = matrix(y - y_Hat4)
qqnorm(error4)
qqline(error4, col="red")
kurtosis(error4)

# Model5
y_Hat5 = m5 %*% thetaHat_5
error5 = matrix(y - y_Hat5)
qqnorm(error5)
qqline(error5, col="red")
kurtosis(error5)


##----Task 2.7:Split the input and output EEG dataset (𝐗 and 𝐲) into two parts: one for training, the for testing----
train<-EEG_dataset[1:141, ]  #training dataset
test<- EEG_dataset[142:201, ] #testing dataset


# 1. Estimate model's parameter on the training set
#y= t_1*x2 + t_2*x1^3 + t_3*x3^4 + tbias + £

one = matrix(1 , nrow(train), 1)  
x_2= matrix(train$x2)  # x2
x_1_cube= matrix(train$x1^3)   # x1 ^ 3
x_3_qud= matrix(train$x3^4)    # x3 ^ 4

X= cbind(x_2, x_1_cube, x_3_qud, one)  #combining variables of the polynomial
y= matrix(train$y)
thetaHat = solve(t(X) %*% X) %*% t(X) %*% y  #estimates the Model's parameters
print(thetaHat)

#2: compute the model’s output/prediction on the testing data.
one = matrix(1 , nrow(test), 1)  
x2= matrix(test$x2)  # x2
x1_cube= matrix(test$x1^3)   # x1 ^ 3
x3_qud= matrix(test$x3^4)    # x3 ^ 4

X_test= cbind(x2, x1_cube, x3_qud, one)  #combining variables of the polynomial
y_Hat = X_test %*% thetaHat
error = y - y_Hat


#3:compute the 95% (model prediction) confidence intervals and plot them (with error bars) 
#together with the model prediction, as well as the testing data samples.

y=matrix(test$y)

n=60
number_of_parameters= 4
sse = norm(error , type = "2")^2  #computes the sum squared error 
sigma_2 = sse/( n - 1 )   #computes the error variance 
cov_thetaHat = sigma_2 * (solve(t(X_test) %*% X_test))
var_y_Hat = matrix(0 , n , 1)

for( i in 1:n){
    X_i = matrix( X_test[i,] , 1 , number_of_parameters ) # this creates a vector matrix.
    var_y_Hat[i,1] = X_i %*% cov_thetaHat %*% t(X_i) 
}

CI = 2 * sqrt(var_y_Hat) #computes confidence interval

plot(x2, y_Hat , col="red")   #plots each variables of the model against output
segments(x2, y_Hat-CI, x2, y_Hat+CI)  #adds erroe bars to each of them.
points(x2, test$y, col="blue", pch="*")

plot(x1_cube, y_Hat , col="red")   #plots each variables of the model against output
segments(x1_cube, y_Hat-CI, x1_cube, y_Hat+CI)
points(x1_cube, test$y, col="blue", pch="*")

plot(x3_qud, y_Hat , col="red")   #plots each variables of the model against output
segments(x3_qud, y_Hat-CI, x3_qud, y_Hat+CI)
points(x3_qud, test$y, col="blue", pch="*")

#--------------------------------------------------
###--------------Task 3: ABC---------
#--------------------------------------------------
set.seed(213)

n <-10000
theta_1<-c()    #Initializing the counter
theta_2 <- c() 

epsilon<- 1.3* RSS3   #assuming this value of epsilon
Y<- EEG_dataset$y


for (i in 1:n) {        #simulating uniform data
    theta1<- runif(1, -5*thetaHat_3[1], 5*thetaHat_3[1])   #setting priors using the 2 largest estimate value from task 2.1
    theta2<- runif(1, -5*thetaHat_3[4], 5*thetaHat_3[4])
    new_t<- thetaHat_3
    new_t[1]<- theta1 
    new_t[4]<- theta2
    y_hat<- m3 %*% new_t
    Rss <- sum((Y-(y_hat))^2)
   
    if (Rss < epsilon) {
      theta_1 <- c(theta_1, theta1)
      theta_2 <- c(theta_2, theta2)
    }
  }

 
plot(theta_1, theta_2)


