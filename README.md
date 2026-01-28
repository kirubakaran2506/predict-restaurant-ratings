Restaurant Rating Prediction using Machine Learning

Project Overview
This project focuses on predicting the aggregate rating of a restaurant using machine learning regression techniques. The model learns from various restaurant-related features such as location, pricing, cuisines, and customer engagement to estimate ratings accurately.
The implementation follows a complete machine learning pipeline, including data preprocessing, feature encoding, model training, evaluation, and interpretation of influential features.

Objective
To build and compare regression models that can predict restaurant ratings and identify the key factors that influence customer ratings.

Technologies Used

Python
Pandas & NumPy – Data handling and preprocessing
Scikit-learn – Machine learning models and evaluation
Linear Regression
Decision Tree Regression

Dataset Overview
The dataset (`Dataset .csv`) contains structured information about restaurants, including:
* Votes
* City and locality
* Latitude and longitude
* Cuisines
* Average cost for two
* Price range
* Online delivery availability
* Table booking availability
* Aggregate rating (target variable)

Methodology
1. Data Preprocessing
* Numerical missing values are filled using the mean
* Categorical missing values are filled using the mode
* Categorical variables are converted into numerical form using **Label Encoding**

2. Feature and Target Selection
Independent variables (X): All features except `Aggregate rating`
Dependent variable (y): `Aggregate rating`

3. Train-Test Split
The dataset is split into **80% training data** and **20% testing data** to evaluate model performance fairly

4. Model Training
Two regression models are trained:

* Linear Regression for baseline performance
* Decision Tree Regression to capture non-linear relationships

5. Model Evaluation
Models are evaluated using:

Mean Squared Error (MSE) – measures prediction error
R-squared (R²) Score– measures goodness of fit

6. Feature Importance Analysis
Decision Tree Regression is used to identify the most influential features affecting restaurant ratings


Results
The Decision Tree Regression model achieved a significantly higher R² score compared to Linear Regression, indicating better prediction accuracy.
Votes emerged as the most influential feature, followed by location-based attributes and cuisine information.

Conclusion
This project demonstrates how machine learning regression techniques can be applied to real-world restaurant data to predict ratings effectively. Feature importance analysis provides valuable insights into factors that influence customer satisfaction and restaurant performance.

Future Enhancements
* Implement ensemble models such as Random Forest or Gradient Boosting
* Perform hyperparameter tuning
* Add visualization of predictions and feature importance
* Deploy the model using a web framework
