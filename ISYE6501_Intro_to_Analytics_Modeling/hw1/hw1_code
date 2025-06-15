# Pre-load packages

# library("sqldf")
# library("RSQLite")
library("ggplot2")
# library("flexdashboard")
# library("plotly")
library("dplyr")
library("plyr")
library("openxlsx")   # Reads, writes, and edits excel files without Java dependency
library("Dict")
library("kernlab")
library("kknn")


##### HW1 Data Prep #####

# sets the file directory to grab data
setwd("C:/Users/Michael/OneDrive - Georgia State University/Desktop/OMSA/ISYE6501_IAM/homework/hw1")

# anchors the filepath
getwd()                                                                                                 

# obtaining credit card with headers data, since space is delimiter, set sep = ""
credit_card_df1 = read.csv("credit_card_data-headers.txt", header = TRUE, sep = "")                     




##### HW1 Question 2.2, Part 1 #####

# convert to matrix
credit_card_matrix_v1 <- as.matrix(credit_card_df1)              


# call ksvm. Vanilladot is a simple linear kernel.
model <- ksvm(credit_card_matrix_v1[,1:10], credit_card_matrix_v1[,11], type="C-svc", kernel=rbfdot(), C=100, scaled=TRUE)
# vanilladot(), rbfdot(), polydot(), tanhdot(), laplacedot(), besseldot(), anovadot(), splinedot()

# let's try different c-values of signficant sizing .00001,.0001,.001,.01,.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000,100000000

# 10000 accuracy starts to go down
# 100000 accuracy is relatively similar to 

# calculate a1…am
a <- colSums(model@xmatrix[[1]] * model@coef[[1]])
#print(a)

# calculate a0
a0 <- model@b
#print(a0)

# see what the model predicts
pred <- predict(model,credit_card_matrix_v1[,1:10])
#print(pred)


# see what fraction of the model’s predictions match the actual classification
sum(pred == credit_card_matrix_v1[,11]) / nrow(credit_card_matrix_v1)



##### HW1 Question 2.2, Part 3 #####

# data manipulation and prep

# Get distinct values in the first column 
distinct_values_credit_card_df1 <- lapply(credit_card_df1, unique)
print(distinct_values_credit_card_df1)


# R1 is the response variable.
# Binary Variables = (A1, A9, A10, A12)
# Continuous variables =  (A2, A3, A8, A11, A14, A15)
# since there are continous variables, there should be standardization present 
# standardization ensures that each feature contributes equally to the distance calculations in kNN. Wihtout it, variables with larger scales skew the distance measure.

credit_card_df2 <- credit_card_df1

cc_predictors <- setdiff(names(credit_card_df2), "R1")

credit_card_df2[cc_predictors] <- scale(credit_card_df2[cc_predictors])

cc_binary_cols <- c("A1","A9","A10","A12")

cc_cont_cols <- setdiff(names(credit_card_df2), c("R1", cc_binary_cols))

credit_card_df2[cc_cont_cols] <- scale(credit_card_df2[cc_cont_cols])

credit_card_df2 <- cbind(credit_card_df2[cc_binary_cols], credit_card_df2[cc_cont_cols], R1 = credit_card_df2$R1)

# Split data into training and testing sets
set.seed(420)
cc_train_indices <- sample(1:nrow(credit_card_df2), size = 0.8 * nrow(credit_card_df2))
cc_train_data <- credit_card_df2[cc_train_indices, ]
cc_test_data <- credit_card_df2[-cc_train_indices, ]

# Function to evaluate different k values
evaluate_k <- function(k) {
  model <- kknn(formula = R1 ~ ., train = cc_train_data, test = cc_test_data, k = k, distance = 2, kernel = "optimal", scale = TRUE)
  pred <- round(fitted(model))
  error_rate <- mean(pred != cc_test_data$R1)
  return(error_rate)
}

# Evaluate error rates for a range of k values
k_values <- 1:20
error_rates <- sapply(k_values, evaluate_k)

# Find the best k value
best_k <- k_values[which.min(error_rates)]

# Plot error rates
plot(k_values, error_rates, type = "b", xlab = "k-value", ylab = "Error Rate", main = "Error Rate vs. k-value")
abline(v = best_k, col = "red", lty = 2)

print(paste("Best k value:", best_k))
