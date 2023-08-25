# A Churn Prediction Model using Ensemble Learning

Our focus will be on a churn dataset, where "Churned Customers" are individuals who have chosen to terminate their association with their current company. XYZ operates as a service provider, offering customers a one-year subscription plan for their product. The company aims to predict whether customers will renew their subscriptions for the upcoming year.

## Table of Contents

- [Description](#description)
- [Architecture](#architecture)
- [Features](#features)
- [Modular_Code_Overview](#modular_code_overview)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Contact](#contact)

## Description

A Decision Tree is a supervised learning method suitable for both classification and regression tasks, often used for solving classification problems. It's structured as a tree-like classifier, with nodes representing dataset features, branches as decision rules, and leaf nodes as outcomes.

This graphical representation offers potential solutions based on given conditions. Beginning from a root node, it grows branches and forms a tree structure. The tree poses questions and branches further based on Yes/No answers.

Our case study focuses on a churn dataset, where "churned customers" are those ending relationships with their current company. XYZ, a service provider, offers a one-year subscription plan and wants to predict customer renewal.

We've previously examined the functionality of the logistic regression model using this dataset in the initial project of this series: [Churn Analysis for Streaming App using Logistic Regression](https://github.com/diegovillatoromx/logistic_regresion_model).

Additionally, we've implemented the decision tree algorithm in our second project: [Building a Customer Churn Prediction Model using Decision Trees](https://github.com/diegovillatoromx/Customer_Churn_Prediction_Model).
It's advisable to review these two projects beforehand as we delve into ensemble techniques.

## Architecture



## Data Description
The CSV consists of around 2000 rows and 16 columns in the [dataset](https://github.com/diegovillatoromx/ensemble_learning/blob/main/input/data_regression.csv)

#### Features:
  - Year
  - Customer_id - unique id
  - Phone_no - customer phone no
  - Gender -Male/Female
  - Age
  - No of days subscribed - the number of days since the subscription
  - Multi-screen - does the customer have a single/ multiple screen subscription
  - Mail subscription - customer receive mails or not
  - Weekly mins watched - number of minutes watched weekly
  - Minimum daily mins - minimum minutes watched
  - Maximum daily mins - maximum minutes watched
  - Weekly nights max mins - number of minutes watched at night time
  - Videos watched - total number of videos watched
  - Maximum_days_inactive - days since inactive
  - Customer support calls - number of customer support calls
  - Churn
      - 0 No
      - 1 Yes 

## Modular_Code_Overview

```
  input
    |_data_regression.csv

  ML_pipeline
    |_evaluate_metrics.py
    |_lime.py
    |_ml_model.py
    |_utils.py

  Tutorial
    |_decision_tree.ipynb

  output
    |_LIME_reports folder
    |_models folder
    |_ROC_curves folder
```
1. Input - It contains all the data that we have for analysis. There is one csv
file in our case:
   - Data_regression.csv
2. ML_Pipeline
   - The ML_pipeline is a folder that contains all the functions put into different
      python files, which are appropriately named. These python functions are
      then called inside the engine.py file.

3. Output
   – Output folder – The output folder contains three subfolders.
     - LIME_reports - contains the LIME reports generated for all three algorithms.
     - Models - contains the models generated for all three algorithms.
     - ROC_curves - contains the ROC curves generated for all three algorithms.
4. Tutorial - This is a reference folder. It contains the [ipython notebook tutorial].

## Installation

Below are the steps required to set up the environment and run this Data Science project on your local machine. Make sure you have the following installed:
- Python 3.x: You can download it from [python.org](https://www.python.org/downloads/).
- Pip: The Python package manager. In most cases, it comes pre-installed with Python. If not, you can install it by following [this guide](https://pip.pypa.io/en/stable/installing/).

### Prerequisites

Install required packages using the requirements.txt file:
``` bash
pip install -r requirements.txt
```
### Installation Steps

1. **Clone the Repository:**

   Clone this repository to your local machine using Git:

   ```bash
   git clone https://github.com/diegovillatoromx/Customer_Churn_Prediction_Model
   cd yourproject
   ```
## Usage

How to utilize and operate the Data Science project after completing the installation steps.
### Data Preparation
Before analysis, prepare data by loading and processing it:
1. ##### Import the required libraries
    ```terminal
    import pickle
    from ML_Pipeline.utils import read_data,inspection,null_values
    from ML_Pipeline.ml_model import prepare_model_smote,run_model
    from ML_Pipeline.evaluate_metrics import confusion_matrix,roc_curve
    from ML_Pipeline.lime import lime_explanation
    import matplotlib.pyplot as plt
    ```
2. ##### Data loading
    If data is in CSV format, load it using Pandas:
    ```terminal
    datapath = 'input/data_regression.csv'
    df = read_data(datapath)
    df.head(5)
    ```
    ![df_head](https://github.com/diegovillatoromx/ensemble_learning/blob/main/images/dfhead.png)
 
3. #### Inspection and cleaning the data
    ```terminal
    x = inspection(df)
    ```
    ![inspection](https://github.com/diegovillatoromx/ensemble_learning/blob/main/images/inspection.png)
4. #### Cleaning and Preprocessing:
   Clean data by handling missing values, normalization, etc.
    ```terminal
    df = null_values(df)
    ```
### Training Model
Perform analysis and modeling on prepared data:

1. #### Model Selection
   Selecting only the numerical columns and excluding the columns we specified in the function
   ```terminal
    X_train, X_test, y_train, y_test = prepare_model_smote(df,class_col='churn',
                                                 cols_to_exclude=['customer_id','phone_no', 'year'])
    ```
### Evaluation

1. #### Evaluation Metrics
   ```terminal
   model_rf,y_pred = run_model('random',X_train,X_test,y_train,y_test)
   ```
   ![running_model](https://github.com/diegovillatoromx/ensemble_learning/blob/main/images/run_model.png)


2. #### Performance metrics
   ```terminal
   conf_matrix = confusion_matrix(y_test,y_pred)
   ```
   ![running_model](https://github.com/diegovillatoromx/ensemble_learning/blob/main/images/cof_matrix.png)

   ```terminal
   import os
   os.makedirs("output/ROC_curves", exist_ok=True)
   roc_val = roc_curve(model_rf, X_test, y_test) # plot the roc curve
   plt.savefig("output/ROC_curves/ROC_Curve_rf.png") # plot the featu
   ```
   ![ROC](https://github.com/diegovillatoromx/ensemble_learning/blob/main/images/Log_ROC.png)

   ```terminal
   os.makedirs("output/models", exist_ok=True)
   pickle.dump(model_rf, open('output/models/model_rf.pkl', 'wb'))
   ```
   
3. #### Feature Importance
   ```terminal
   os.makedirs("output/LIME_reports", exist_ok=True)
   lime_exp = lime_explanation(model_rf,X_train,X_test,['Not Churn','Churn'],1)
   lime_exp.savefig('output/LIME_reports/lime_report_rf.jpg')   ```
   ![running_model](https://github.com/diegovillatoromx/ensemble_learning/blob/main/images/lime_repor_rf.png)


# Contributing

Encourage contributions to the Data Science project.

## How to Contribute
### Fork the Repository:
Fork this repository to your GitHub account.

### Clone Your Fork:
Clone your fork to your local machine:
```
git clone https://github.com/yourusername/yourproject.git
cd yourproject
```
### Create a Branch:
Create a new branch to work on:
```
git checkout -b your-branch-name
```
### Make Changes:
Make necessary changes to the code.

### Testing and Feedback:
Test changes and seek feedback from collaborators.

### Submit a Pull Request:
Send a pull request from your branch to the main repository.

## Contribution Guidelines
1. Focus changes on specific improvements.
2. Follow project's coding style.
3. Provide detailed descriptions in pull requests.
## Reporting Issues
Use "Issues" to report bugs or suggest improvements.

# Contact
For questions or contact, [email](diegovillatormx@gmail.com) or [Twitter](https://twitter.com/diegovillatomx) .
