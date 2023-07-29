# Deep Learning Challenge: Predicting Success of Nonprofit Organizations
## Overview
In this project, I worked on a deep learning model to help the nonprofit foundation, Alphabet Soup, in selecting applicants for funding with the best chance of success in their ventures. I utilized machine learning techniques and neural networks to build a binary classifier that predicts whether applicants will be successful if funded by Alphabet Soup.

## Data Preprocessing
- Target Variable: The target variable for our model is IS_SUCCESSFUL, which indicates whether the money provided to the organization was used effectively (1 for Yes, 0 for No).

- Feature Variables: The feature variables for our model include columns such as APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, and ASK_AMT.

- Removed Columns: I dropped the EIN and NAME columns as they are identification columns and not relevant for our prediction.

- Unique Values: I determined the number of unique values for each column to get insights into the data distribution.

- Binning: For columns with more than 10 unique values, I binned the rare categorical variables together in a new value, "Other," to reduce the number of unique categories.

- Encoding: I used pd.get_dummies() to encode the categorical variables into a numerical format for use in the model.

- Train-Test Split: The preprocessed data were split into features array X and target array y, followed by splitting the data into training and testing datasets using train_test_split.

- Scaling: The training and testing features datasets were scaled using StandardScaler to normalize the data.

## Model Design and Evaluation
Neural Network Model: I designed a deep neural network with two hidden layers. The first hidden layer has 80 nodes with a 'relu' activation function, and the second hidden layer has 30 nodes with a 'relu' activation function. The output layer has 1 node with a 'sigmoid' activation function for binary classification.

Model Compilation and Training: The model was compiled with 'binary_crossentropy' as the loss function, 'adam' optimizer, and 'accuracy' as the evaluation metric. It was then trained using the training data.

Model Evaluation: I evaluated the model using the test data to calculate the loss and accuracy.

## Model Optimization
In an attempt to achieve a target predictive accuracy higher than 75%, I made multiple iterations and optimizations to the model. I tried different combinations of hidden layers, nodes, activation functions, and epochs.

## Results
* Data Preprocessing
Target Variable: IS_SUCCESSFUL
Feature Variables: APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT
Removed Columns: EIN, NAME
* Compiling, Training, and Evaluating the Model
* Neurons and Layers: The final model has two hidden layers with 80 and 30 nodes, respectively, and an output layer with 1 node.

* Activation Functions: Hidden layers use 'relu' activation, and the output layer uses 'sigmoid' activation.

* Model Performance: The model achieved an accuracy of approximately XX%, but it did not reach the target accuracy of 75%.

* Steps to Increase Performance: To improve model performance, I experimented with different activation functions, and additional hidden layers, and adjusted the number of epochs.

## Summary
The deep learning model showed promising results, but it did not achieve the desired predictive accuracy of 75% or higher. Further optimization may be required, including trying different architectures, hyperparameter tuning, and feature engineering. Additionally, alternative models like random forests or gradient boosting could be explored to solve this classification problem.

In conclusion, the deep learning model provided valuable insights into predicting the success of nonprofit organizations, and with additional enhancements, it can potentially become a robust tool for Alphabet Soup's funding decision-making process.
