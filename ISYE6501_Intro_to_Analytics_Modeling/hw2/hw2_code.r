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
library("caret")


##### HW1 Data Prep #####

# sets the file directory to grab data
setwd("C:/Users/Michael/OneDrive - Georgia State University/Desktop/OMSA/ISYE6501_IAM/homework/hw1")

# anchors the filepath
getwd()                                                                                                 

# obtaining credit card with headers data, since space is delimiter, set sep = ""
credit_card_df1 = read.csv("credit_card_data-headers.txt", header = TRUE, sep = "")


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


# try 5, 10, 20, 50, 100 folds
# Function to evaluate different k values with cross-validation for a given number of folds
# Function to evaluate different k values with cross-validation for a given number of folds
evaluate_k_cv_folds <- function(k, num_folds) {
  set.seed(420)
  folds <- createFolds(credit_card_df2$R1, k = num_folds)
  error_rates <- c()
  
  for (i in 1:num_folds) {
    fold_indices <- folds[[i]]
    cv_train_data <- credit_card_df2[-fold_indices, ]
    cv_test_data <- credit_card_df2[fold_indices, ]
    
    model <- kknn(formula = R1 ~ ., train = cv_train_data, test = cv_test_data, k = k, distance = 2, kernel = "optimal", scale = TRUE)
    pred <- round(fitted(model))
    error_rate <- mean(pred != cv_test_data$R1)
    error_rates <- c(error_rates, error_rate)
  }
  
  return(error_rates)  # Return the vector of error rates instead of the mean
}

# Initialize an empty data frame to store results
results <- data.frame()

# Evaluate error rates for a range of k values and different numbers of folds
k_values <- 1:20
fold_numbers <- c(5, 10, 20, 50, 100)

for (num_folds in fold_numbers) {
  for (k in k_values) {
    error_rates_cv <- evaluate_k_cv_folds(k, num_folds)
    avg_error_rate <- mean(error_rates_cv)
    variance <- sd(error_rates_cv)
    results <- rbind(results, data.frame(folds = num_folds, k = k, avg_error_rate = avg_error_rate, variance = variance))
  }
}


# Plot Bias (Average Error Rate) and Variance
ggplot(results, aes(x = folds)) +
  geom_line(aes(y = avg_error_rate, color = "Bias (Avg Error Rate)")) +
  geom_point(aes(y = avg_error_rate, color = "Bias (Avg Error Rate)")) +
  geom_line(aes(y = variance, color = "Variance (SD of Error Rates)")) +
  geom_point(aes(y = variance, color = "Variance (SD of Error Rates)")) +
  labs(title = "Bias and Variance Estimates Over Different k-Fold Numbers",
       x = "Number of Folds",
       y = "Estimate",
       color = "Metric") +
  theme_minimal()

print(results_summary)

# Summarize results to find the best k value for each fold number
best_k_results <- results %>%
  group_by(folds) %>%
  filter(avg_error_rate == min(avg_error_rate)) %>%
  select(folds, k, avg_error_rate)

# Plot Best k Values vs Error Rate
ggplot(best_k_results, aes(x = k, y = avg_error_rate, color = factor(folds))) +
  geom_line() +
  geom_point() +
  labs(title = "Best k Values vs Error Rate",
       x = "k-Value",
       y = "Average Error Rate",
       color = "Number of Folds") +
  theme_minimal()

print(results)
print(best_k_results)

# # # # # # # # # # # # # Question 4.2 

# sets the file directory to grab data
setwd("C:/Users/Michael/OneDrive - Georgia State University/Desktop/OMSA/ISYE6501_IAM/homework/hw2")

# anchors the filepath
getwd()                                                                                                 

# obtaining iris data, since space is delimiter, set sep = ""
iris_df1 = read.csv("iris.txt", header = TRUE, sep = "")

#Interaction terms

# iris_df1$Sepal_Ratio <- iris_df1$Sepal.Length / iris_df1$Sepal.Width
# iris_df1$Petal_Product <- iris_df1$Petal.Length * iris_df1$Petal.Width

# iris_df1$Sepal_Petal_Difference <- iris_df1$Sepal.Length - iris_df1$Petal.Length
# iris_df1$Sepal_Petal_Sum <- iris_df1$Sepal.Length + iris_df1$Petal.Length


# scale the data
iris_df1[, 1:4] <- scale(iris_df1[, 1:4])
# all_iris_features <- cbind(iris_df1[, 1:4], iris_df1[, 6:9])  # Combine original predictors and interaction features
# all_iris_features_scaled <- scale(all_features)


#check for na's
sum(is.na(iris_df1))

# correlation plot
# cor(iris_df1[, 1:4])
# cor(iris_df1[, 1:4], iris_df1[, 6:9])

# Set seed for reproducibility
set.seed(117)

# Function to calculate total within-cluster sum of squares for different k values
wss <- function(k) {
  kmeans(iris_df1[, c(1:4)], k, nstart = 10)$tot.withinss
}

wss <- function(k) {
  kmeans(iris_df1[, c(1:4, 6:9)], k, nstart = 10)$tot.withinss
}

# Number of clusters
k.values <- 1:10

# Calculate wss for each k
wss.values <- sapply(k.values, wss)

# Plot the elbow method
plot(k.values, wss.values, type = "b", pch = 19, frame = FALSE,
     xlab = "Number of clusters K",
     ylab = "Total within-clusters sum of squares")


# Perform k-means clustering with k = 3
set.seed(117)
kmeans_result <- kmeans(iris_df1[, c(1:4)], centers = 3, nstart = 10)
kmeans_result <- kmeans(iris_df1[, c(1:4, 6:9)], centers = 3, nstart = 10)

# Add the cluster assignments to the dataframe
iris_df1$Cluster <- as.factor(kmeans_result$cluster)

# View the clustering results
table(iris_df1$Species, iris_df1$Cluster)


# Create the confusion matrix
confusion_matrix <- matrix(c(50, 0, 0,
                             0, 39, 11,
                             0, 14, 36), 
                           nrow = 3, byrow = TRUE,
                           dimnames = list(c("setosa", "versicolor", "virginica"),
                                           c("Cluster 1", "Cluster 2", "Cluster 3")))

# Number of correctly clustered samples
correct_setosa <- confusion_matrix["setosa", "Cluster 1"]
correct_versicolor <- confusion_matrix["versicolor", "Cluster 2"]
correct_virginica <- confusion_matrix["virginica", "Cluster 3"]

# Total number of correctly clustered samples
total_correct <- correct_setosa + correct_versicolor + correct_virginica

# Total number of samples
total_samples <- sum(confusion_matrix)

# Overall accuracy
overall_accuracy <- total_correct / total_samples

# Print the overall accuracy
overall_accuracy
