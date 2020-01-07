#https://www.analyticsvidhya.com/blog/2017/06/a-comprehensive-guide-for-linear-ridge-and-lasso-regression/
#https://www.rdocumentation.org/packages/glmnet/versions/3.0-1/topics/glmnet

library(dplyr)
library(stargazer)
library(caret)
library(readr)
library(boot)
library(caTools)
library(e1071)
library(glmnet)
library(kernlab)
library(ROCR)

tit_train = read.csv(file="tit-train-f.csv", header=TRUE, sep=",")
anyNA(tit_train) #no NAs
summary(tit_train)
str(tit_train) #look at class of each variable




####### ridge logistic regression #######
#https://towardsdatascience.com/ridge-regression-for-better-usage-2f19b3a202db
#https://www.mailman.columbia.edu/research/population-health-methods/ridge-regression

#ridge regression is a variant of linear regression
#iterate certain values onto the lambda and evaluate the model with a measurement such as ‘Mean Square Error (MSE)’. So, the lambda value that minimizes MSE should be selected as the final model. 
#if we increase the lambda value, ridge β’s (coeffs) should decrease. But ridge β’s can’t be zeros no matter how big the lambda value is set. That is, ridge regression gives different importance weights to the features but does not drop unimportant features.
#Since ridge regression shrinks coefficients by penalizing, the features should be scaled for start condition to be fair. 
#Note that the coefficients at lambda equal to zero (x = 0) are the same with the OLS (linear regression: ordinary least squares) coefficients.
#The value of lambda determines how much the ridge parameters differ from the parameters obtained using OLS, and it can take on any value greater than or equal to 0. When lambda=0, this is equivalent to using OLS
#an OLS model with some bias (favors one indep (more important feature) var over the other) is better at prediction than the pure OLS model, we call this modified OLS model as the ridge regression model.
#there is a cost to this decrease in variance: an increase in bias. However, the bias introduced by ridge regression is almost always toward the null. Thus, ridge regression is considered a “shrinkage method”, since it typically shrinks the beta coefficients toward 0.
#there is always a value of lambda>0 such that the mean square error (MSE) is smaller than the MSE obtained using OLS. However, determining the ideal value of lambda is impossible, because it ultimately depends on the unknown parameters. Thus, the ideal value of lambda can only be estimated from the data.
#Cross validation simply entails looking at subsets of data and calculating the coefficient estimates for each subset of data, using the same value of lambda across subsets. This is then repeated multiple times with different values of lambda. The value of lambda that minimizes the differences in coefficient estimates across these data subsets is then selected.






#find best lambda in ridge using grid
#grid = 10^seq(10,-2,length=100) #for finding best lambda using grid versus cross val
#ridge.mod1 = glmnet(x,y,alpha=0,lambda=grid,family="binomial") #alpha 0 is for ridge regression
#ridge.mod=glmnet(x,y,alpha=0,lambda=grid, familyt="binomial") #for binary outcome
#dim(coef(ridge.mod))
#ridge.mod$lambda[50] #looking at a lambda of 50 and its coefficient
#as lambda goes up, shrinkage penalty grows
#best lambda is trade off between bias and variance so find optimal
#coef(ridge.mod)[,50]
#sqrt(sum(coef(ridge.mod)[-1,50]^2)) #looks at squared error
#compute error for each lambda and choose smallest error
#look at mean square after changing lambda aka s
#ridge.mod$lambda[60] #looking at a lambda of 60
#coef(ridge.mod)[,60]
#sqrt(sum(coef(ridge.mod)[-1,60]^2))
#predict(ridge.mod,s=50,type="coefficients")[1:20,] #s is best lambda
#ridge does not go to 0. If you remove any pred vars, remove ones that are very low and close to 0

#set.seed(1)
#train = sample(1:nrow(x), nrow(x)/2)
#test = (-train)
#y.test = y[test]
# alpha = 0 is ridge regression
#ridge.mod2 = glmnet(x[train,],y[train],alpha=0,lambda=grid, thresh=1e-12, family="binomial")
#ridge.pred = predict(ridge.mod,s=4,newx=x[test,])
#mean((ridge.pred-y.test)^2)
#mean((mean(y[train])-y.test)^2)
#ridge.pred = predict(ridge.mod,s=1e10,newx=x[test,])
#mean((ridge.pred-y.test)^2)

# when lambda =0, it is the same as linear regression
#ridge.pred = predict(ridge.mod,s=0,newx=x[test,],exact=T,x=x[train,],y=y[train])
#mean((ridge.pred-y.test)^2)
#lm(y~x, subset=train)
#predict(ridge.mod,s=0,exact=T,type="coefficients",x=x[train,],y=y[train])[1:20,]

############# RIDGE using cross validation instead of grid to find best lambda
set.seed(1)
x = model.matrix(Survived~.,tit_train)[,-1] #sets up x with all vars except for target survived
head(x)
colnames(x)
y = tit_train$Survived #sets up y with target var only
train=sample(1:nrow(x), nrow(x)/2)
test=(-train)
y.test=y[test]

cv.out = cv.glmnet(x[train,],y[train],alpha=0, family="binomial")
summary(cv.out)
#gives you # of lambdas looked at, mean, standard deviation, cvup and cvlo for conf intervl
plot(cv.out)
#best model under lambd.min
#with this, look 1 standard error from lowest point on graph for best lambda
bestlam = cv.out$lambda.min
bestlam
#gives you best lambda to choose
#0.02565232 is best lambda

#mean((ridge.pred-y.test)^2) #not used for an output that is factored
out = glmnet(x,y,alpha=0, family="binomial")
predict(out,type="coefficients",s=bestlam)[1:13,] #13 is var total including target
#exponent on 10 is negative, I am looking for a small number. If the exponent is a seven for example, I will be moving the decimal point seven places. Since I need to move the point to get a small number, I'll be moving it to the left.

#PassengerId -0.0002530132 #Pclass -0.5435723  #Sex -2.161273 #Age -0.01336017
#SibSp -0.2135848 #Parch -0.02209793 #Ticket 0.00000008885684 #Fare 0.002802999
#Cabin 0.6226431 #Embarked -0.1978456 #name 0.0005413021 #title 0.1491452 

##Taking out vars closest to 0: PassengerId, Ticket, Fare, name 
#???could take out as are closer to 0: Age, Parch, SibSp, Cabin, Embarked


####### lasso regression #######
#lasso used to select pred variables to use in ml methods

set.seed(1)
cv.out_lasso = cv.glmnet(x[train,],y[train],alpha=1, family="binomial")
#alpha 1 is lasso
cv.out_lasso
#model output tells you min lambda using mse (mean squared error)
summary(cv.out_lasso)
plot(cv.out_lasso)
# the best model is under lambda = lambda.min
bestlam2 = cv.out_lasso$lambda.min
bestlam2
#0.004696268 best lambda for lasso
out2 = glmnet(x,y, alpha=1, family = "binomial")
plot(out2, xvar = "lambda", label = TRUE)
lasso.coef = predict(out2,type="coefficients",s=bestlam)[1:13,]
lasso.coef
#take out 0 coefficients for lasso, 0 vars did not explain anything in model
lasso.coef[lasso.coef!=0]
#selected variables with coefficients not 0
#Pclass, Sex, Age, SibSp, Cabin, title


####### logisitc regression with cross val #######

#could also use cv.glm( ) from boot package
#https://www.r-bloggers.com/evaluating-logistic-regression-models/
#http://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html#log



#define training control
train_control = trainControl(method = "cv", number = 10)

#train the model 
logist_model = train(Survived ~ .,
               data = tit_train,
               trControl = train_control,
               method = "glm",
               family=binomial())

summary(logist_model)
#significant pred vars
  #Pclass      -7.541e-01  1.630e-01  -4.627 3.71e-06 ***
  #Sex         -2.661e+00  2.091e-01 -12.723  < 2e-16 ***
  #Age         -1.956e-02  5.546e-03  -3.528 0.000419 ***
  #SibSp       -2.913e-01  1.060e-01  -2.748 0.005991 ** 
  #Cabin        6.699e-01  2.617e-01   2.560 0.010478 *
  #title        1.433e-01  5.650e-02   2.537 0.011192 *

logist_model$results
#accuracy is 0.793583

#ctrl = trainControl(method = "repeatedcv", number = 10, savePredictions = TRUE)

#mod_fit_again = train(Survived ~ .,  data=tit_train, 
                     #method="glm", family="binomial",
                     #trControl = ctrl, tuneLength = 5)
#summary(mod_fit_again)
#same results
#pred = predict(mod_fit_again, newdata=testing)
#confusionMatrix(data=pred, testing$Survived)

#http://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html#log
#cvfit = cv.glmnet(x, y, family = "binomial", type.measure = "class")
#cvfit2 = cv.glmnet(x, y, family = "binomial", type.measure = "mae")
#cvfit3 = cv.glmnet(x, y, family = "binomial", type.measure = "auc")
#plot(cvfit)
#cvfit$lambda.min
#cvfit$lambda.1se
#coef(cvfit, s = "lambda.min")
#predict(cvfit, newx = x[1:10,], s = "lambda.min", type = "class")




#test_x dataset that matches length of train dataset
#test = data[-train_split, ]
#test_x_noClass = test[-1]
#another way to split data using caret package
#trainIndex = createDataPartition(titanic$Survived, p=.8, list=F)
#train2 = titanic[trainIndex, ] #use in train function for model creation
#test2 = titanic[-trainIndex, ] #use in predict function 
#test_with_y2 = titanic[-train_split, ]$Survived









####### random forest using lasso selected variables #######

#lasso selected vars Pclass, Sex, Age, SibSp, Cabin, title
#load test data
#test_data = read.csv(file="tit-test-f.csv", header =TRUE, sep=",")
#View(test_data)
#anyNA(test_data)
#summary(test_data)
#str(test_data)

#preprocessing test data
#test_data = test_data[ -c(1) ] #take out id column
#test_data = test_data[ -c(5) ] #take out parch column
#test_data = test_data[ -c(5) ] #take out ticket column
#test_data = test_data[ -c(5) ] #take out fare column
#test_data = test_data[ -c(6) ] #take out embarked column
#test_data = test_data[ -c(6) ] #take out name column
#View(test_data)

#preprocessing train data
data = tit_train[ -c(1) ] #take out id column
data = data[ -c(6) ] #take out parch column
data = data[ -c(6) ] #take out ticket column
data = data[ -c(6) ] #take out fare column
data = data[ -c(7) ] #take out embarked column
data = data[ -c(7) ] #take out name column
View(data)
class= data$Survived


#split train data into train/test how Dr. Cao did in lab 5 example
set.seed(101)
train_split = sample(1:nrow(data), nrow(data)*0.8) #take 80% sample
train = data[train_split, ]
test_without_y = subset(data[-train_split, ], select= -Survived)
test_with_y = data[-train_split, ]$Survived

library(randomForest)
set.seed(101)
control_rand_for = trainControl(method="repeatedcv", number=10, repeats=3)
metric = "Accuracy"
n= round(sqrt(ncol(data)))
tunegrid = expand.grid(.mtry=seq(by = 1,to = 5, from = 1))
rand_for = train(Survived ~ ., data=train, method="rf",
                 metric=metric, tuneGrid= tunegrid, trControl=control_rand_for)
print(rand_for)
#best mtry was 2: accuracy = 0.8394327 and kappa = 0.6522835

pred_rf_caret = predict(rand_for, test_without_y)

confusionMatrix(pred_rf_caret, test_with_y)
#dim(test_data) #418 rows
#dim(data) #891 rows



important = varImp(rand_for) #sgows important variables
important


#for loop to determin best number of trees using ntree
accu = rep(0,1000)
for (i in 1:1000) {
  set.seed(i)
  rand_for = randomForest(Survived~ ., data=train, ntree=i)
  pred_loop = predict(rand_for, test_without_y)
  cm = confusionMatrix(pred_loop, test_with_y)
  
  accu[i] = cm[3]$overall[1]
}

accu
max(accu)
#max accuracy is 0.8156425
#42 trees 


#using 42 trees and now determining best mtry
control_rand_for42 = trainControl(method="repeatedcv", number=10, repeats=3)
metric = "Accuracy"
n= round(sqrt(ncol(train)))
tunegrid = expand.grid(.mtry=seq(by = 1,to = 5, from = 1))
rand_for42 = train(Survived ~ ., data=train, method="rf", ntree=42,
                   metric=metric, tuneGrid= tunegrid, trControl=control_rand_for42)
print(rand_for42)
#mtry = 2 is best with 42 trees

#best model created and tested
rf_model_best = randomForest(Survived~ ., data=train, ntree=42, mtry=2)
rf_model_best

pred_rf_best = predict(rf_model_best, test_without_y)
confusionMatrix(pred_rf_best, test_with_y)


####### logistic regression #######
data2 = data
data2$Pclass = factor(data2$Pclass)
data2$Sex = factor(data2$Sex)
train_split2 = sample(1:nrow(data2), nrow(data2)*0.8) #take 80% sample
train2 = data2[train_split2, ]
test_without_y2 = subset(data2[-train_split2, ], select= -Survived)
test_with_y2 = data2[-train_split2, ]$Survived


tb_pclass = table(data2$Survived, data2$Pclass)
tb_pclass
chisq.test(tb_pclass, correct=FALSE) #INDEPENDENT TEST, ARE THESE ASSOCIATED?
#p is smaller than .05 so reject null hypothesis but don't accept alternative yet

tb_sex = table(data2$Survived, data2$Sex)
tb_sex
chisq.test(tb_sex, correct=FALSE) #INDEPENDENT TEST, ARE THESE ASSOCIATED?
#p is smaller than .05 so reject null hypothesis but don't accept alternative yet

tb_sib = table(data2$Survived, data2$SibSp)
tb_sib

tb_cabin = table(data2$Survived, data2$Cabin)
tb_cabin

tb_title = table(data2$Survived, data2$title)
tb_title

pairs(data2) #to see if any input variables are highly correlated. If so, take out.




set.seed(101)
logist_mod = glm(Survived ~ Pclass+Sex+Age+SibSp+Cabin+title, 
                 data=train2, family=binomial(link="logit"))
summary(logist_mod)
exp(coef(logist_mod))

pred_log =predict(logist_mod, newdata = test_without_y2, type = "response")
surv_or_not = ifelse(pred_log >= 0.5, "YES", "NO")
surv_or_not
#1 is positive survived and 0 is did not survive
#convert to factor: fac_class3
fac_class = factor(surv_or_not, levels = levels(test_with_y2["Survived"]))

confusionMatrix(fac_class, test_with_y2)

####### svm #######

y = train$Survived
x = subset(train, select =-Survived)

poly_mod = train(x, y, method="svmPoly", allowParallel = FALSE, tuneLength=5,
                 trControl=trainControl(method="repeatedcv", 
                                        number=5, repeats=3))
poly_mod #to see performance
### BEST MODEL ###
#The final values used for the model were degree = 3, scale = 0.1 and C = 4.
#accuracy = 0.8328638 and kappa = 0.63156825

linea_mod = train(x, y, method="svmLinear", tuneLength=5,
                  trControl=trainControl(method="repeatedcv", 
                                         number=5, repeats=3))
linea_mod
#accuracy = 0.791198
#kappa = 0.5531621
#Tuning parameter 'C' was held constant at a value of 1

radial_mod = train(x, y, method="svmRadial", allowParallel = FALSE, tuneLength=5,
                   trControl=trainControl(method="repeatedcv", 
                                          number=5, repeats=3))
radial_mod
#final values used for the model were sigma = 0.3381143 and C = 1.
#accuracy = 0.8295709 
#kappa = 0.6276866
#tuning parameter 'sigma' was held constant at a value of 0.3381143

pred_supp = predict(poly_mod, test_without_y)
#y2 = subset(data.f, split==FALSE)$V10
confusionMatrix(pred_supp, test_with_y)

#bootstrap 100 samples and calculate the 95% CI and AUC using poly kernal
n = 100
accuracy = rep(0,n)
auc = rep(0,n)
for (i in 1:n) {
  set.seed(i+100)
  new_index = sample(c(1:length(data$Survived)), length(data$Survived), replace=TRUE)
  new_sample = data[new_index,]
  split = sample.split(new_sample$Survived, SplitRatio = 0.8)
  train = subset(new_sample, split==TRUE)
  test.x = subset(subset(new_sample, split==FALSE), select=-Survived)
  test.y = subset(new_sample, split==FALSE)$Survived
  
  svm_poly = train(Survived~ ., data=train, method="svmPoly",
                   tuneLength = 1,
                   trControl= trainControl(method="repeatedcv",
                                           repeats = 5, 
                                           classProbs=TRUE))
  
  #accuracy
  pred = predict(svm_poly, test.x)
  c = confusionMatrix(pred, test.y)
  accuracy[i] = c[3]$overall[1]
  
  #auc
  pred = predict(svm_poly, test.x, type="prob")[,2]
  out = prediction(pred, test.y)
  auc[i] = performance(out, measure="auc")@y.values[[1]]
}

#95% ci for accuracy
accuracy
accuracy.mean = mean(accuracy)
accuracy.mean
#accuracy mean = 0.7601309
accuracy.me = qnorm(0.975) * sd(accuracy)/sqrt(length(accuracy)) #Divide your accuracy standard deviation by the square root of your accuracy size.
accuracy.me
accuracy.lci = accuracy.mean - accuracy.me #lower end of ci
accuracy.lci
accuracy.uci = accuracy.mean + accuracy.me #upper end of ci
accuracy.uci
#lower ci = 0.7538999
#upper ci = 0.7663618

#95% ci for auc
auc.mean = mean(auc)
auc.me = qnorm(0.975) * sd(auc)/sqrt(length(auc))
auc.lci = auc.mean - auc.me
auc.uci = auc.mean + auc.me
auc.lci
auc.uci
#lower auc ci 0.8394324
#upper auc ci 0.8520922