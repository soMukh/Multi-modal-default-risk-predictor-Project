# Loan Approval Case Study

## 📌 Project Overview
This project analyzes a **Loan Approval dataset** to predict whether a loan application should be approved. It involves **data preprocessing, exploratory data analysis (EDA), feature engineering, and machine learning models** to compare different classifiers.

The goal is to build a predictive model that helps financial institutions make better lending decisions.

---

## 📂 Dataset
The dataset contains loan application details such as:
- Applicant demographics (Gender, Education, Dependents, Married)
- Financial details (ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History)
- Loan application status (Approved: `Y` / Rejected: `N`)

---

## 🛠️ Steps in the Notebook
1. **Data Preprocessing**
   - Handling missing values (e.g., LoanAmount, Dependents).
   - Encoding categorical variables.
   - Normalization/standardization of numeric features.

2. **Exploratory Data Analysis (EDA)**
   - Visualizations of income, loan amounts, and approval status.
   - Distribution analysis of categorical variables.
   - Correlation heatmap for numerical features.

3. **Model Training & Evaluation**
   
   Implemented machine learning models:
   - **K-Nearest Neighbors (KNN)** → Hyperparameter tuning on `k`.
   - **Support Vector Machine (SVM)** → Tested with different `C` values.
   - **Naive Bayes (GaussianNB)** → Probabilistic baseline model.
   - **Decision Tree** → Tuned using `max_depth`.
   - **Random Forest (Bagging)** → Tuned with different numbers of estimators (`n_estimators`).

   Each model was evaluated using **accuracy score** on the test set.

5. **Model Comparison**
   - Accuracy results of all models were compared.
   - Best-performing model identified for loan approval prediction.

---

## 📊 Results
- **KNN**: Accuracy depends on the choice of `k`.  
- **SVM**: Strong performance with optimized `C`.  
- **Naive Bayes**: Simple and fast but lower accuracy compared to others.  
- **Decision Tree**: Good interpretability but risk of overfitting.  
- **Random Forest**: Generally the most robust and highest accuracy.  

---

## 🚀 How to Run
1. Clone the repository / download the notebook.  
2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
3. Open Jupyter Notebook:
   ```bash
   jupyter notebook Loan_approval_case_study.ipynb
   ```
4. Run all cells to see preprocessing, analysis, and results.
