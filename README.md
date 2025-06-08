# Customer Churn Prediction for a Telecom Company

## 1. Project Overview

This project is an end-to-end data science project focused on predicting customer churn for a fictional telecom company. The goal is to build a machine learning model that can accurately identify customers who are likely to leave the company. By identifying these customers proactively, the business can offer targeted incentives and interventions to improve customer retention and reduce revenue loss.

This project demonstrates a complete machine learning workflow, including:
- In-depth Exploratory Data Analysis (EDA)
- Feature Engineering and Data Preprocessing
- Model Training and Hyperparameter Tuning
- Model Evaluation and Interpretation
- Final Model Selection and Saving

---

## 2. Business Problem

Customer churn is a critical metric for any subscription-based business. The cost of acquiring a new customer is often significantly higher than retaining an existing one. This project aims to solve this problem by building a classification model that predicts a binary outcome: whether a customer will `Churn` (Yes) or `Not Churn` (No).

---

## 3. Data Source

The dataset used is the "Telco Customer Churn" dataset, sourced from Kaggle. It contains customer information, including demographics, subscribed services, account information, and the target churn label.

---

## 4. Methodology

The project followed a structured machine learning workflow:

1.  **Data Exploration & Visualization (EDA):** The initial phase involved a deep dive into the dataset to understand feature distributions, correlations, and initial patterns related to churn. Key insights were uncovered using libraries like `pandas`, `matplotlib`, and `seaborn`.
2.  **Data Preprocessing:** The data was cleaned and transformed to be suitable for modeling. This included handling missing values, encoding categorical features using One-Hot Encoding, and scaling numerical features using `StandardScaler`.
3.  **Model Building & Training:** Several classification models were trained and evaluated:
    - **Logistic Regression:** Served as a strong, interpretable baseline model.
    - **Random Forest Classifier:** A more complex ensemble model tested with default settings, class balancing, and finally, systematic hyperparameter tuning.
4.  **Model Evaluation:** Models were evaluated based on a suite of metrics, with a strong focus on **Recall** and **F1-Score** for the "Churn" class, as identifying potential churners is the key business goal.
5.  **Model Interpretation:** The winning model was analyzed to understand which features were the most significant drivers of churn.

---

## 5. Model Performance and Key Findings

### Final Model Selection

After rigorous evaluation, the **Logistic Regression** model was chosen as the final, champion model. While more complex models like Random Forest were tuned extensively, the simpler linear model provided the best performance on the key business metrics, especially Recall.

| Metric | Logistic Regression | Tuned Random Forest | **Winner** |
| :--- | :--- | :--- | :--- |
| **Accuracy**| **0.8204** | 0.8126 | Logistic Regression |
| **Recall (Churn)** | **0.60** | 0.51 | **Logistic Regression** |
| **F1-Score (Churn)**| **0.64** | 0.59 | **Logistic Regression** |

This demonstrates a key real-world finding: model complexity does not guarantee superior performance. A well-prepared dataset with a robust baseline can be the most effective solution.

### Key Churn Drivers (from Logistic Regression)
- **Contract Type:** Customers on a `Month-to-month` contract are far more likely to churn.
- **Internet Service:** Customers with `Fiber optic` service show a higher propensity to churn.
- **Tenure:** Shorter tenure is one of the strongest predictors of churn. Long-term customers are very loyal.

---

## 6. How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/customer-churn-predictor.git](https://github.com/your-username/customer-churn-predictor.git)
    cd customer-churn-predictor
    ```
2.  **Set up the Conda environment:**
    ```bash
    conda create --name churn-project python=3.10
    conda activate churn-project
    ```
3.  **Install dependencies:**
    ```bash
    pip install notebook pandas matplotlib seaborn scikit-learn joblib pyarrow
    ```
4.  **Run the notebooks:**
    - Open and run the cells in `01_Data_Exploration.ipynb` to see the analysis and save the processed data.
    - Open and run the cells in `02_Model_Training.ipynb` to train, evaluate, and save the final model.

---

## 7. Future Improvements

While this project delivers a valuable baseline model, future iterations could explore:
- **Advanced Models:** Implementing and tuning Gradient Boosting models like XGBoost or LightGBM.
- **Feature Engineering:** Creating new, interaction-based features (e.g., ratio of tenure to monthly charges).
- **Advanced Imbalance Handling:** Using techniques like SMOTE to create synthetic data for the minority class.
- **Deployment:** Packaging the final model and scaler into a simple API using a web framework like Flask or FastAPI.
