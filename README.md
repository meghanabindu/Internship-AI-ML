"Machine Learning Model Collection"
This repository provides a beginner-to-intermediate level demonstration of several popular machine learning algorithms implemented using Python and scikit-learn. The models include:

Linear Regression (House Price Prediction)

Logistic Regression (Diabetes Prediction & Titanic Survival)

Support Vector Machines (Digit Recognition)

K-Nearest Neighbors (Custom & Iris Dataset)


1. Linear Regression – House Price Prediction
Objective: Predict house prices based on area (in square feet).

Model Used: LinearRegression

Metrics: Mean Squared Error (MSE), R² Score

Result:
MSE ≈ 114,856
R² ≈ 0.997 (Excellent fit)

Visualization: Plots the regression line over actual data points using matplotlib.

2. Logistic Regression – Diabetes Classification
Dataset: Pima Indians Diabetes Dataset
(source)

Features: Pregnancies, Glucose, Blood Pressure, etc.

Target: Binary outcome (Diabetic or not)

Model: LogisticRegression

Accuracy: ~74.7%

Metrics: Confusion matrix and classification report included.

3. Logistic Regression – Titanic Survival Prediction
Dataset: Titanic Dataset from Seaborn

Features: Passenger class, sex, age, fare

Target: Survival (0 = No, 1 = Yes)

Model: LogisticRegression

Accuracy: ~75.5%

Extra: Includes data visualization and a confusion matrix heatmap using Seaborn.

4. Support Vector Machine – Handwritten Digit Recognition
Dataset: load_digits() from sklearn

Objective: Classify digits (0–9) based on pixel values.

Model: SVC(kernel='rbf')

Interactive: Allows user to input a test sample number to check prediction vs. actual.

5. Custom KNN Implementation – Simple Class Prediction
A manual implementation of the K-Nearest Neighbors (KNN) algorithm using Euclidean distance and Counter for majority vote.

Input: Custom training data with 2D features

Output: Predicted class for a test sample

Algorithm: Written from scratch (No scikit-learn)

6. KNN Classifier – Iris Flower Classification
Dataset: Iris flower dataset (load_iris())

Model: KNeighborsClassifier(n_neighbors=3)

Accuracy: ~95.5%

Use Case: Classic example for multi-class classification.


