# Student Dropout Prediction

## Problem Statement
Predict whether a higher education student is at risk of 
dropping out before it actually happens.

## Dataset
- Source: Kaggle — Predict Students Dropout and Academic Success
- Records: 4,424 students
- Features: 34 columns
- Target: Dropout / Non-Dropout (Binary)

## Project Steps
1. Data Understanding
2. EDA + Visualization
3. Preprocessing
4. Model Building — 8 models before and after SMOTE
5. Hyperparameter Tuning — RandomizedSearchCV
6. Feature Importance
7. Streamlit Deployment

## Models Used
- Logistic Regression
- Decision Tree
- Random Forest ← Best Model
- AdaBoost
- Gradient Boosting
- XGBoost
- KNN
- Gaussian Naive Bayes

## Key Findings
- 2nd semester grades and approved units are strongest dropout predictors
- Students with unpaid tuition fees have high dropout risk
- Academic performance is stronger predictor than financial support

## How to Run
1. Install dependencies
   pip install -r requirements.txt

2. Run notebook to generate model
   Open studentdropoutprediction.ipynb and run all cells

3. Run Streamlit app
   streamlit run app.py

## Tools Used
Python, Pandas, Scikit-learn, XGBoost, SMOTE, 
Streamlit, Matplotlib, Seaborn