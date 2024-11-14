library(mlflow)
library(caret)
library(e1071)
library(datasets)  # for the iris dataset
library(dplyr)     # for data wrangling

# Load the iris dataset
data(iris)

# Split the data into training and test sets
set.seed(42)
train_indices <- createDataPartition(iris$Species, p = 0.8, list = FALSE)
train_data <- iris[train_indices, ]
test_data <- iris[-train_indices, ]

# Define the model hyperparameters
params <- list(
  solver = "lbfgs",
  max_iter = 1000,
  multi_class = "auto",
  random_state = 8888
)

# Train the Logistic Regression model
model <- train(Species ~ ., data = train_data, method = "multinom", trace = FALSE)

# Predict on the test set
predictions <- predict(model, newdata = test_data)

# Calculate accuracy
accuracy <- sum(predictions == test_data$Species) / nrow(test_data)

# Using Databricks
mlflow_set_tracking_uri("databricks")
mlflow_set_tracking_uri(uri="http://127.0.0.1:8080")
experiment_path <- "/Users/m.h.maruf@tue.nl/quickstart"

# Create a new MLflow experiment
mlflow_set_experiment(experiment_path)
# Start an MLflow run
mlflow_start_run()

# Log the hyperparameters
# mlflow_log_params(params) # causes error, there's only mlflow_log_param in R API

# Log the hyperparameters
mlflow_log_param("solver", params$solver)
mlflow_log_param("max_iter", params$max_iter)

# Log the accuracy metric
mlflow_log_metric("accuracy", accuracy)

# Set a tag for this run
mlflow_set_tag("Training Info", "Basic LR model for iris data")

# Log the model
mlflow_log_model(model, artifact_path = "iris_model")

mlflow_end_run()

# Output the results
result <- cbind(test_data, predicted_class = predictions)
print(head(result, 4))
