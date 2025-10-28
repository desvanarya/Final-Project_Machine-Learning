🚗 Car Price Prediction App
Predicting Used Car Prices with Machine Learning using Streamlit
This Streamlit web app predicts the selling price of used cars based on various features such as mileage, car age, fuel type, transmission, and brand tier.
The model is built using Ridge Regression, ensuring robust and consistent performance.

🧾 Overview
This project demonstrates how data preprocessing, feature engineering, and regression models can be combined to predict used car prices.
It includes:
A user-friendly web interface (built with Streamlit)
Interactive EDA visualizations
A trained machine learning model
Evaluation metrics for transparency and validation

✨ Features
✅ Predict car prices instantly based on user input
✅ Display correlation heatmaps and price distributions
✅ Show real-time model evaluation (MAE, MSE, RMSE, R²)
✅ Clean and interactive Streamlit web interface
✅ Automated feature encoding and preprocessing

📊 Dataset
The dataset used is sourced from Car Dekho Dataset (Kaggle).
Main columns include:
year – Year of manufacture
selling_price – Price of the car
km_driven – Distance driven
fuel – Fuel type (Petrol, Diesel, CNG, etc.)
seller_type – Type of seller (Dealer, Individual)
transmission – Transmission type (Manual, Automatic)
owner – Ownership status
name – Brand and model name

🧠 Machine Learning Model
This project uses Ridge Regression, a regularized linear regression model that minimizes overfitting by penalizing large coefficients.
Pipeline Overview:
Data cleaning and removal of duplicates
Extracting car brands and mapping them to tiers (High, Mid, Low, Other)
Creating a new feature — car_age from year
One-hot encoding of categorical columns
Splitting data into train (80%) and test (20%) sets
Training the Ridge Regression model
Evaluating the model with regression metrics

📈 Exploratory Data Analysis (EDA)
Interactive EDA is built directly into the Streamlit app:
Correlation Heatmap: Shows relationships between numerical features
Price Distribution Plot: Displays the spread of car prices in the dataset

