# Lending Club Loan Data Analysis

A deep learning project for predicting loan default risk using neural networks on Lending Club loan data.

## Overview

This project implements a Deep Neural Network (DNN) model to predict whether a borrower will fully pay back their loan or default. The model uses historical loan data from Lending Club, including borrower credit information, loan details, and purpose of the loan.

## Dataset

The project uses a loan dataset (`loan_data.csv`) with the following key features:

- **credit.policy**: Whether the borrower meets the credit underwriting criteria
- **int.rate**: The interest rate of the loan
- **installment**: The monthly installments owed by the borrower
- **log.annual.inc**: The natural log of the self-reported annual income
- **dti**: Debt-to-income ratio
- **fico**: FICO credit score
- **days.with.cr.line**: Number of days the borrower has had a credit line
- **revol.bal**: Revolver balance (amount unpaid at the end of credit card billing cycle)
- **revol.util**: Revolving line utilization rate
- **inq.last.6mths**: Number of inquiries in the last 6 months
- **delinq.2yrs**: Number of times the borrower had been 30+ days past due
- **pub.rec**: Number of derogatory public records
- **purpose**: Purpose of the loan (categorical feature)
- **not.fully.paid**: Target variable (1 = borrower will not fully pay the loan, 0 = will fully pay)

## Project Structure

```
.
├── Lending_Club_Loan_Data_Analysis_Guilherme_Faria.ipynb  # Main analysis notebook
├── loan_data.csv                                          # Dataset (not included)
└── README.md                                              # This file
```

## Installation

### Prerequisites

- Python 3.7+
- Jupyter Notebook

### Required Packages

Install the required dependencies:

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
```

Or use a requirements.txt file:

```bash
pip install -r requirements.txt
```

## Usage

1. **Place the dataset**: Ensure `loan_data.csv` is in the same directory as the notebook.

2. **Run the notebook**: Open and execute the Jupyter notebook:

```bash
jupyter notebook Lending_Club_Loan_Data_Analysis_Guilherme_Faria.ipynb
```

3. **Follow the analysis steps**:
   - Data loading and preprocessing
   - Exploratory Data Analysis (EDA)
   - Feature engineering and transformation
   - Model training
   - Evaluation and results

## Methodology

### 1. Data Preprocessing
- Removed missing values and duplicates
- Applied One-Hot Encoding to the categorical `purpose` feature
- Used StandardScaler to normalize numerical features

### 2. Exploratory Data Analysis
- Analyzed target variable distribution (16% default rate)
- Visualized class imbalance
- Performed correlation analysis to identify multicollinearity

### 3. Feature Engineering
- Removed highly correlated features (correlation > 0.80) to reduce multicollinearity
- Final dataset: 18 features after transformation

### 4. Model Architecture

Built a Deep Neural Network using Keras/TensorFlow with the following structure:

```
Input Layer (128 neurons, ReLU activation)
    ↓ Dropout (0.3)
Hidden Layer 1 (64 neurons, ReLU activation)
    ↓ Dropout (0.3)
Hidden Layer 2 (32 neurons, ReLU activation)
    ↓ Dropout (0.2)
Output Layer (1 neuron, Sigmoid activation)
```

**Model Parameters:**
- Total parameters: 12,801
- Optimizer: Adam (learning rate = 0.001)
- Loss function: Binary Crossentropy
- Batch size: 64
- Epochs: 50
- Validation split: 10%

### 5. Evaluation Metrics

Due to the imbalanced nature of the dataset, the following metrics were used:
- **AUC-ROC Score**: Primary metric for imbalanced classification
- Classification Report (precision, recall, F1-score)
- Confusion Matrix

## Results

### Model Performance
- **AUC-ROC Score**: 0.6612
- **Accuracy**: 84%
- **True Negatives**: 2,389
- **False Positives**: 25
- **False Negatives**: 444
- **True Positives**: 16

### Observations
- The model achieves high accuracy but struggles with the minority class (defaulters)
- Low recall for the positive class (3%) indicates the model misses many actual defaults
- The AUC score of 0.66 suggests moderate discriminative ability
- The class imbalance (16% default rate) significantly impacts model performance

## Potential Improvements

1. **Address Class Imbalance**:
   - Use SMOTE (Synthetic Minority Over-sampling Technique)
   - Apply class weights during training
   - Try undersampling the majority class

2. **Model Optimization**:
   - Hyperparameter tuning (learning rate, batch size, number of layers)
   - Try different architectures (deeper networks, different activation functions)
   - Implement early stopping to prevent overfitting

3. **Feature Engineering**:
   - Create new features based on domain knowledge
   - Perform more sophisticated feature selection
   - Consider interaction terms between features

4. **Alternative Models**:
   - Compare with ensemble methods (Random Forest, XGBoost)
   - Try traditional ML models (Logistic Regression, SVM)

## Author

**Guilherme Faria**

Deep Learning Project - SimpliLearn

## License

This project is for educational purposes.

## Acknowledgments

- Dataset: Lending Club
- Frameworks: TensorFlow/Keras, Scikit-learn, Pandas, NumPy
- Visualization: Matplotlib, Seaborn
