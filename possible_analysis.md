# Notes from the lab that could be usefull for the project ananlysis

## Lab 1: First Steps in Linear Regression
1. pairs(dataset)
2. round(cor(dataset), digits = 2)
3. summary(mod)

## Lab 2: Multiple Linear Regression
1. dataset$column <- factor(dataset$column)
2. levels(dataset$column) <- c("No", "Yes", "Etc.")
3. new_mod <- update(old_mod, . ~ . + new_predictor)
4. monet$new_column <- with(dataset, column1 * column2)
5. library(car) residualPlots(mod)
6. qqPlot(residuals(mod3))
7. log-transformation of the response, since we have noted that again the minimum of the fitted value is negative
8. function I(column / 1000) used to rescale some column
9. Compare models with the R^2 on the same scale otherwise the comaprison is not fair
10. which.min(fitted(mod)) -> to see minimum of the fitteed model make sens or not, we can investigate also the maximum
11. predict(mod, newdata = data.frame(colum1 = value1, column2 = value2)) -> for the non scaled model
12. exp(predict(mod, newdata = data.frame(colum1 = value1, column2 = value2))) -> for the log-scaled model
13. mod <- update(old_mod, . ~ . + log(column) : factorial_column)

## Lab 3: Multiple Linear Regression / part 2
1. dataset$column <- factor(dataset$column, levels = c("Low", "Medium", "High"))
2. dataset <- read.csv("../data/dataset.csv", stringsAsFactors = TRUE)
3. table(dataset$column)
4. summary(exp(predict(mod))) -> for the log transforation
5. vif(mod) -> Multicollinearity is the occurrence of high intercorrelations among two or more independent variables in a multiple regression model
6. plot(allEffects(mod)) -> Effect plots display the predicted response as a function of one predictor at turn with the values of other predictors fixed at their mean values. Function **allEffects** from package **effects** computes all the effects in a mode


## Lab 4: Polynomial Regression
1. mod_new <- update(mod_old, . ~ . + I(column ^ 2))
2. 
```{r}
## split the graphical device in two columns
par(mfrow = c(1, 2))
## left panel: linear model
plot(log(price) ~ carat, data = diamond)
abline(modA, col = "blue", lwd = 1.5)
## right panel: quadratic model
plot(log(price) ~ carat, data = diamond)
## model prediction
predB <- predict(modB)
## get the indices of the sorted carat
ix <- sort(diamond$carat, index.return = TRUE)$ix
## plot sorted predictions
lines(diamond$carat[ix], predB[ix], col = "red", lwd = 1.5)
```
3. 
```{r}
plot(log(price) ~ carat, data = diamond)
abline(modA, col = "blue", lwd = 1.5)
lines(diamond$carat[ix], predB[ix], col = "red", lwd = 1.5)
```
4. modC.poly <- lm(log(price) ~ poly(carat, degree = 3), data = diamond)
5. The estimated model coefficients are different because of the orthogonal polynomial representation, but **modC.poly** and **modC** are the same fitted model

## Lab 5: Interaction Terms
1. boxplot(Salary ~ Gender, data = gender, col = c("darkorange", "steelblue")) -> plot the boxplot for the given relation
2. with(gender, by(Salary, Gender, summary)) -> with is used to apply some function to the first and second colum that produce a relation

## Lab 6: Influential Points
1. influenceIndexPlot(mod, vars = "Cook") -> identify the influential point
2. compareCoefs(mod1, mod2, pvals = TRUE) -> allows to compare the estimated coefficients and their standard errors for the two fitted models, pvals allows to add the p-values
3. mod_new <- update(mod_old, subset = -c(7, 18, 33)) -> remove the specified observation by idx

## Lab 7: Power Transformations
1. powerTransform(mod) -> suggests to take the log-transformation, we take the log transformation because the estimated value of **lambda** is very close to zero


## Lab 8: Logistic Regression
1. dataset <- na.omit(dataset) -> removes all the rows with missing values
2. mod <- glm(respose ~ predictor + predicotr + ..., data = dataset, 
family = binomial) -> logistic regression, famili = binomial needed
3. Effect plots for logistic regression are displayed in the log-odds scale but labelled on the response scale (here the probability of survival). It is also possible to display effect plots directly on the response scale using argument **rescale.axis = FALSE**
4. p1 <- predict(mod3, newdata = data.frame(Pclass = "1", Sex = "female", Age = 30))
exp(p1) / (1 + exp(p1)) -> inverse logit transformation
5. or diretly -> predict(mod3, newdata = data.frame(Pclass = "1", Sex = "female", Age = 30), type = "response")
6. predict can be used also to compute the prediction interval, we have to be shure to do the inverse lgit in order to be on the probability scale


## Lab 9: Classification
### Logistic Regression
1. confuzion matrix
```{r}
preds50 <- glm.probs > 0.5
table(preds = preds50, true = titanic.test$Survived)
```
2. glm.roc <- roc(titanic.test$Survived ~ glm.probs, plot = TRUE, print.auc = TRUE) -> area under the curve
3. coords(glm.roc, x = "best", ret = "all") -> allows to extract the coordinates of the ROC curve at a specific point or at the `best point' corresponding to the maximum of the sum of sensitivity and specificity

### Linear Discriminant Analysis
1. lda.fit <- lda(Survived ~ Pclass + Sex + Age, data = titanic.train)
2. print(with(titanic.train, mean(Pclass[Survived == "TRUE"] == 1))) -> group mean
3. table(preds = lda.preds$class, true = titanic.test$Survived) -> confusion matrix with the 50% threshold
4. lda.preds <- predict(lda.fit, newdata = titanic.test) -> predict for the test data
5. The components are the predicted **class**, the **posterior** class probability and the linear discriminants **x**. The first two components are those of major interest. **class** returnt eh predicted calss and **posterior** return the respective probability distribution for that observation
6. The predicted class is computed with the 50% threshold
7. table(preds = lda.preds$class, true = titanic.test$Survived) -> the confusion matrix with the 50% threshold
8. lda.roc <- roc(titanic.test$Survived ~ lda.preds$posterior[, 2], plot = TRUE, print.auc = TRUE) -> roc curve for discriminant analysis

### Quadratic Discriminant Analysis
12. qda.fit <- qda(Survived ~ Pclass + Sex + Age, data = titanic.train) -> quadraic discriminant analysis

### Naive Bayes
1. nb.fit <- naiveBayes(Survived ~ Pclass + Sex + Age, data = titanic.train) -> naive bayes

### K-Nearest Neighbours
1. x.train <- model.matrix(~ Pclass + Sex + Age, data = titanic.train) -> matrix of predictiors with categorical variable coded as dummy variables
2. x.test <- model.matrix(~ Pclass + Sex + Age, data = titanic.test)[, -1] -> same but excluding the response
3. preds.knn <- knn(train = x.train, test = x.test, cl = titanic.train$Survived, k = 5) -> apply knn


## Lab 10: Classification - part 2
### Multinomial Logistic Regression
1. multi.fit <- multinom(Pclass ~ Survived + Sex + Age, data = titanic.train, trace = FALSE) -> multinomial logistic regression. Option trace = FALSE avoids printing updates from the optimizer employed by multinom
2. summary(multi.fit, Wald.ratios = TRUE) -> The model summary reports only the estimated coefficients and their standard errors. Option **Wald.ratios** adds the Z statistics also know as Wald ratio statistic
3. The summary does not report the p-values that are, however, readily computed from the Wald ratios.
4. table(preds = multi.preds <- predict(multi.fit, newdata = titanic.test), true = titanic.test$Pclass) -> the agreement between predictions and test data can be summarized with a confusion matrix
5. mean(multi.preds == titanic.test$Pclass) -> accuray of multinomial logistic regression

### Linear and Quadratic Discriminant Analysis
1. **Linear:** qda.fit <- qda(Pclass ~ Survived + Sex + Age, data = titanic.train)
2. **Quadratic:** qda.fit <- qda(Pclass ~ Survived + Sex + Age, data = titanic.train)

### Naive Bayes
1. nb.fit <- naiveBayes(Pclass ~ Survived + Sex + Age, data = titanic.train)

### K-Nearest Neighbours
```{r}
x.train <- model.matrix(~ Survived + Sex + Age, data = titanic.train)[, -1]
x.test <- model.matrix(~ Survived + Sex + Age, data = titanic.test)[, -1]
library("class")
set.seed(98765)
rates <- double(50)
for (i in 1:50) {
  tmp <- knn(train = x.train, test = x.test, cl = titanic.train$Pclass, k = i)
  rates[i] <- mean(tmp == titanic.test$Pclass)
}
plot(x = (1:50), y = rates, xlab = "k", ylab = "Accuracy", type = "l")
```

## Lab 11: Cross-Validation
### Leave-One-Out Cross-Validation
1. Cross-validation for linear and generalized linear models can be carried out using function cv.glm from package boot.
2. A call to glm with family gaussian (default choice) gives the same results of a call to lm
3. cv.err <- cv.glm(Auto, glm.fit) -> slot 'delta' of 'cv.err' gives the cross-validated mean square error:

### K-Fold Cross-Validation
```{r}
cv.error.10 <- double(10)
for (i in 1:10){
    glm.fit <- glm(mpg ~ poly(horsepower, i), data = Auto)
    cv.error.10[i] <- cv.glm(Auto, glm.fit, K = 10)$delta[1]
}
cv.error.10
```

### Comparison of different models that uses LOO-CV and KF-CV
1. Using a list of models makes the analysis easier to run (and read!). Create a list with the five fitted models
model.list <- list(mod1, mod2, mod3, mod4, mod5)
2. Now run cv.glm on the models list with a for loop:
```{r}
## one-out
all.cv <- double(5)
for(i in 1:5) {
    all.cv[i] <- cv.glm(titanic, model.list[[i]])$delta[1]
}
all.cv

## ten-fold
all.cv10 <- double(5)
for(i in 1:5) {
    all.cv10[i] <- cv.glm(titanic, model.list[[i]], K = 10)$delta[1]
}
all.cv10
```
3. Create a summary table of the cross-validation results using function cbind (“column bind”)
cbind(all.cv, all.cv10)
4. Modify row and column names
```{r}
rownames(results) <- c("mod1", "mod2", "mod3", "mod4", "mod5")
colnames(results) <- c("one-out", "ten-fold")
results
```

## Lab 12: Subset Selection and Stepwise Regression
### Best Subset Selection
### Stepwise Regression

## Lab 13: Shrinkage Estimation
### Ridge Regression, Lasso, Shrinkage in Classification
interessante da capire se lasso e ridge penaly si possono utilizzare in problemi multiclass con variabili factorial
```{r}
# needs to do the model.matrix however

fit_ridge <- cv.glmnet(X, Y, family = "multinomial", alpha = 0)  # alpha = 0 for Ridge

fit_ridge <- cv.glmnet(X, Y, family = "multinomial", alpha = 1)  # alpha = 1 for Lasso
```


## Lab 14: Shrinkage Estimation - Part 2
```{r}
lasso.formula <- glmnet(diabetes ~ ., data = diabetes.data, family = "binomial", alpha = 1)
coef(lasso.formula, s = best.lambda)

predict(lasso.formula, newdata = diabetes.data[50, ], s = best.lambda, 
    type = "response")
```

## Lab 15: Shrinkage Estimation - part 3
The selected models by glmnet and glmnetUtils are the same, however the output of glmnetUtils contains both the two levels of the categorical predictor Europe whereas the output of glmnet contains only the level TRUE of Europe.

In fact, glmnetUtils deliberately avoids the usual treatment of factors with a reference level (here Europe=FALSE) “absorbed” by the intercept. The reason for this choice is explained in the “vignette” of glmnetUtils

In the above example, glmnet and glmnetUtils identify the same model because the categorical predictor Europe is not relevant. Obviously, there could be substantial differences between the results obtained with the two packages when there are significant categorical predictors.

## Lab 16: Principal Component Analysis

## Lab 17: Visualization of Principal Components

## Lab 18: Dimensionality Reduction Models

## Lab 19: Nonlinear Regression
