\# Cardiovascular Disease Prediction 🫀



\## Overview

This project predicts the presence of cardiovascular disease using patient health records.  

It demonstrates the full machine learning workflow:

\- Data preprocessing \& cleaning

\- Exploratory Data Analysis (EDA)

\- Correlation analysis

\- Model training and comparison

\- Best model selection \& evaluation

\- Saving the trained model for future use



\## Dataset

\- \*\*Source\*\*: Provided dataset (`cardio\_train.csv`) with ~70,000 patient records.

\- \*\*Target Variable\*\*: `cardio` (0 = No disease, 1 = Disease).

\- \*\*Features\*\*:

&nbsp; - Demographics: Age (in years), Gender, Height, Weight, BMI

&nbsp; - Clinical: Systolic/Diastolic BP (`ap\_hi`, `ap\_lo`)

&nbsp; - Lab: Cholesterol, Glucose

&nbsp; - Lifestyle: Smoking, Alcohol intake, Physical activity



\## Methodology

1\. \*\*Preprocessing\*\*

&nbsp;  - Converted `age` from days → years

&nbsp;  - Engineered BMI from height \& weight

&nbsp;  - Cleaned unrealistic values (height, weight, BP, BMI)

&nbsp;  - Handled duplicate rows



2\. \*\*EDA\*\*

&nbsp;  - Distribution plots for numerical features

&nbsp;  - Count plots for categorical features

&nbsp;  - Correlation heatmap

&nbsp;  - Class balance visualization



3\. \*\*Model Training\*\*

&nbsp;  - Logistic Regression

&nbsp;  - Decision Tree

&nbsp;  - K-Nearest Neighbors (KNN)

&nbsp;  - Support Vector Machine (SVM)

&nbsp;  - Random Forest



4\. \*\*Evaluation\*\*

&nbsp;  - Metrics: Accuracy, Precision, Recall, F1-score

&nbsp;  - Confusion matrix

&nbsp;  - Feature importance (Random Forest)



5\. \*\*Best Model\*\*

&nbsp;  - Random Forest achieved the highest accuracy

&nbsp;  - Model saved using Joblib



\## Results

\- Outputs are saved in the `outputs/` folder:

&nbsp; - Plots (EDA, correlation, confusion matrix, feature importance)

&nbsp; - Model performance comparison (`05\_model\_performance.csv`)

&nbsp; - Classification report (`06\_classification\_report.txt`)

&nbsp; - Best trained model (`best\_model\_RandomForest.joblib`)



\## How to Run

1\. Install dependencies:

&nbsp;  ```bash

&nbsp;  pip install numpy pandas matplotlib seaborn scikit-learn joblib



