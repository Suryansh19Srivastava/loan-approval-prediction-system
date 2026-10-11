# Loan Approval Prediction System Using Machine Learning

## Introduction

The Loan Approval Prediction System is a machine learning project developed as part of the AI and ML Summer Internship Program 2026 at the Innovation and Incubation Hub MNNIT Foundation, MNNIT Allahabad, Prayagraj.

The purpose of this project is to predict whether a loan application is likely to be approved or rejected based on the applicant's personal, financial, and credit history information.

The project uses the Kaggle Loan Prediction dataset and a Logistic Regression model. A web application built using Flask allows users to enter applicant details and view the prediction along with the estimated approval probability.

## Problem Statement

Loan approval involves evaluating several factors, including income, employment status, credit history, loan amount, and repayment term. Evaluating these details manually can take time.

This project explores how machine learning can be used to analyze historical loan applications and predict the likely outcome for a new application.

The system is intended for learning and demonstration purposes. Its predictions should not be used as the sole basis for actual lending decisions.

## Objectives

- Understand and preprocess a real-world loan prediction dataset.
- Handle missing values and categorical features.
- Perform exploratory data analysis to identify patterns in the data.
- Apply feature engineering to improve the data used for training.
- Train a Logistic Regression classification model.
- Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
- Integrate the trained model into a Flask web application.
- Display the prediction and probability through a simple interface.

## Dataset

The project uses the Kaggle Loan Prediction dataset, which contains 614 loan application records and 13 original columns, including the loan identifier and target variable.

The model uses the following 11 input features:

| Feature | Description |
|---|---|
| Gender | Gender of the applicant |
| Married | Marital status |
| Dependents | Number of dependents |
| Education | Education level |
| Self_Employed | Employment status |
| ApplicantIncome | Income of the primary applicant |
| CoapplicantIncome | Income contributed by the co-applicant |
| LoanAmount | Requested loan amount |
| Loan_Amount_Term | Loan repayment period in months |
| Credit_History | Historical credit compliance indicator |
| Property_Area | Property location category |

The target variable is `Loan_Status`:

- `1` represents an approved loan.
- `0` represents a rejected loan.

According to the project report, the dataset contains 422 approved applications and 192 rejected applications.

## Methodology

The project follows these steps:

### 1. Data Preprocessing

The dataset is examined for missing values and inconsistencies. The project report describes using mode imputation for categorical features and median imputation for selected numerical features.

Categorical values are converted into numerical representations so that they can be used by the machine learning model.

### 2. Exploratory Data Analysis

Exploratory data analysis is performed to understand the dataset and the relationships between different features.

The analysis includes:

- Applicant income distribution
- Loan amount distribution and outliers
- Distribution of approved and rejected applications
- Relationship between credit history and loan status
- Correlation between selected features

These analyses help identify patterns in the historical data and understand the characteristics of loan applications.

### 3. Feature Engineering

The project report describes applying a natural logarithmic transformation to applicant income to reduce the effect of its right-skewed distribution.

The transformation is:

`LogApplicantIncome = ln(ApplicantIncome + 1)`

### 4. Model Training

Logistic Regression is used as the primary classification algorithm.

It estimates the probability of a loan application belonging to the approved class. The model then produces a classification based on its decision rule.

The dataset is divided into training and testing subsets so that performance can be evaluated on records not used to fit the model.

### 5. Model Evaluation

The trained model is evaluated using accuracy, precision, recall, F1-score, and a confusion matrix.

These metrics provide information about how well the model identifies approved and rejected applications, rather than relying only on overall accuracy.

### 6. Web Application

The trained model is integrated into a Flask application. Users can enter applicant details through a web form, submit the information, and view the predicted outcome.

The application loads the saved model using Joblib and uses NumPy to prepare the input data for prediction.

## Model Performance

The project report records an accuracy of 78.86% on a test set of 123 records.

The reported confusion matrix is:

| Actual / Predicted | Rejected | Approved |
|---|---:|---:|
| Rejected | 18 | 25 |
| Approved | 1 | 79 |

The classification results show that the model identifies most approved applications correctly. However, it also incorrectly predicts approval for some applications that were rejected in the test data.

This highlights the importance of evaluating class-specific precision and recall when working with loan approval data.

These results are based on the evaluation described in the project report and may vary if the model is retrained using a different data split or configuration.

## Features of the Application

- Form for entering applicant information
- Prediction of loan approval or rejection
- Model-generated approval probability
- Result page displaying the prediction
- Flask backend connected to a saved machine learning model
- Separate templates and static files for the user interface

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and model evaluation |
| Logistic Regression | Classification algorithm |
| Joblib | Saving and loading the trained model |
| Flask | Backend web framework |
| HTML | Web page structure |
| CSS | Styling |
| JavaScript | Frontend interactions |
| Jupyter Notebook | Model development and data analysis |

## Project Structure

```text
loan-approval-prediction-system/
│
├── dataset/
├── documentation/
│   ├── PPT.pdf
│   └── Project Report.pdf
├── model/
│   └── loan_model.pkl
├── notebook/
├── static/
├── templates/
│   ├── index.html
│   ├── form.html
│   └── result.html
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

The `dataset` directory contains the data used in the project. The `notebook` directory contains the model development and analysis work. The trained model is stored in `model`, while `app.py` handles the Flask application. The `templates` and `static` directories contain the frontend resources.

## Installation and Setup

### Prerequisites

- Python 3.x
- Git
- A web browser

### Step 1: Clone the Repository

```bash
git clone https://github.com/Suryansh19Srivastava/loan-approval-prediction-system.git
```

### Step 2: Open the Project Directory

```bash
cd loan-approval-prediction-system
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the Application

```bash
python app.py
```

Open the local URL displayed in the terminal. By default, Flask commonly uses:

`http://127.0.0.1:5000/`

Make sure the saved model file is present at `model/loan_model.pkl` before starting the application.

## How to Use

1. Open the application in your browser.
2. Navigate to the loan eligibility form.
3. Enter the applicant's details.
4. Submit the form.
5. View the predicted loan status and the probability displayed on the result page.

The output represents a prediction made by the trained model using historical data. It does not guarantee that a real financial institution will approve or reject the application.

## Limitations

- The model depends on the quality and representativeness of the historical dataset.
- The dataset contains more approved applications than rejected applications.
- Some rejected applications are incorrectly predicted as approved.
- The model considers a limited set of applicant features.
- Predicted probabilities may not correspond to actual approval rates in real-world lending.
- The project has not been established as suitable for production financial decisions.

## Future Scope

The project could be extended in several ways:

- Experiment with additional classification algorithms.
- Apply cross-validation and hyperparameter tuning.
- Improve evaluation of the minority class.
- Add explainability techniques to understand individual predictions.
- Improve input validation and error handling.
- Package preprocessing and prediction into a consistent machine learning pipeline.
- Add secure storage and tracking of submitted applications.
- Evaluate the model on additional, relevant datasets.

## Project Documentation

The `documentation` directory contains the project report and presentation:

- `Project Report.pdf` describes the problem, methodology, data analysis, model evaluation, and future scope.
- `PPT.pdf` contains the project presentation.

## Author

**Suryansh Srivastava**

B.Tech in Computer Science Engineering  
United College of Engineering and Research, Prayagraj

GitHub: [Suryansh19Srivastava](https://github.com/Suryansh19Srivastava)

Project Repository: [Loan Approval Prediction System](https://github.com/Suryansh19Srivastava/loan-approval-prediction-system)

## Acknowledgements

This project was developed as part of the AI and ML Summer Internship Program 2026 at the Innovation and Incubation Hub MNNIT Foundation, MNNIT Allahabad, Prayagraj.

The project uses the Kaggle Loan Prediction dataset for educational and experimental purposes.
