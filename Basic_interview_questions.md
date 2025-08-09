# 📄 MLOps / AI-ML Workflow & Interview Q&A

## Table of Contents
1. [Data Understanding](#1-data-understanding)
2. [Data Preparation](#2-data-preparation)
3. [Feature Engineering](#3-feature-engineering)
4. [Feature Importance](#4-feature-importance)
5. [Feature Selection](#5-feature-selection)
6. [Training](#6-training)
7. [Parameter & Hyperparameter Tuning](#7-parameter--hyperparameter-tuning)
8. [Validation](#8-validation)
9. [Model Maturity](#9-model-maturity)
10. [Data Drift & Model Drift](#10-data-drift--model-drift)
11. [Additional Interview Q&A](#11-additional-interview-qa)

---

## 1. Data Understanding
**Q:** What is data understanding in the ML lifecycle?  
**A:** Data understanding is the process of exploring and analyzing datasets to gain insights about their structure, quality, and relevance to the problem.

**Q:** Why is it important?  
**A:** Helps identify data quality issues early, guides feature engineering decisions, and ensures the data aligns with the business problem.

**Techniques & Tools:**  
- Pandas Profiling, Sweetviz, Databricks Data Explorer  
- Summary statistics  
- Visualization: histograms, box plots, scatter plots  

---

## 2. Data Preparation
**Q:** What does data preparation involve?  
**A:** Cleaning and transforming raw data into a usable format, handling missing values, correcting data types, scaling, and encoding categorical variables.

**Common Challenges:**  
- Missing values  
- Inconsistent formats  
- Duplicate records  
- Unbalanced datasets  

**Techniques:**  
- Imputation (mean, median, mode, KNN)  
- Standardization/Normalization  
- One-hot encoding, label encoding  
- Removing duplicates  

---

## 3. Feature Engineering
**Q:** What is feature engineering?  
**A:** Creating or modifying features to improve model performance.

**Examples:**  
- Extracting text sentiment  
- Binning continuous values  
- Date transformations (e.g., days since event)

---

## 4. Feature Importance
**Q:** What is feature importance?  
**A:** A measure of how much each feature contributes to the predictive power of a model. It tells what features are important for the training.

**Methods:**  
- Model-based importance (RandomForest, XGBoost)  
- Permutation importance  
- SHAP values  

---

## 5. Feature Selection
**Q:** Why perform feature selection?  
**A:** Reduce dimensionality, improve generalization, speed up training, and avoid overfitting.

**Methods:**  
- Filter: Correlation, Chi-square  
- Wrapper: Recursive Feature Elimination (RFE)  
- Embedded: LASSO, tree-based selection  

---

## 6. Training
**Q:** What happens during training?  
**A:** The model learns patterns from data by minimizing a loss function.

**Best Practices:**  
- Train on representative data  
- Use pipelines for reproducibility  
- Version control with MLflow  

---

## 7. Parameter & Hyperparameter Tuning
**Q:** Difference?  
- **Parameters:** Learned during training (e.g., model weights).  
- **Hyperparameters:** Set before training (e.g., learning rate, max depth).  

**Tuning Methods:**  
- Grid Search  
- Random Search  
- Bayesian Optimization  
- Hyperopt / Optuna  

---

## 8. Validation
**Q:** What is validation?  
**A:** Evaluating model performance on unseen data.

**Techniques:**  
- Train/Test Split  
- K-Fold Cross-Validation  
- Stratified Sampling  

**Metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC, RMSE  

---

## 9. Model Maturity
**Q:** What is model maturity?  
**A:** A measure of readiness for production deployment.

**Stages:**  
1. **POC:** Experimental  
2. **Staging:** Tested and validated  
3. **Production:** Deployed with monitoring  
4. **Retired:** No longer used  

---

## 10. Data Drift & Model Drift
**Q:** What is data drift?  
**A:** Change in feature distributions compared to training data.

**Q:** What is model drift?  
**A:** Decrease in performance of model due to changes in data patterns. Decrease in the model performance metrics.

**Detection:**  
- Accuracy monitoring  
- Kolmogorov–Smirnov test  
- Tools: EvidentlyAI, WhyLabs  

**Mitigation:**  
- Retraining  
- Feature recalibration  

---

## 11. Additional Interview Q&A

### 11.1 Model Overfit vs Underfit
| Aspect | Overfitting | Underfitting |
|--------|------------|--------------|
| Definition | Learns patterns + noise, poor generalization | Fails to learn patterns |
| Cause | Too complex model | Too simple model |
| Symptom | High train acc, low test acc | Low train & test acc |
| Fix | Regularization, reduce complexity | Increase complexity |

---

### 11.2 MLOps vs LLMOps
| Feature | MLOps | LLMOps |
|---------|-------|--------|
| Scope | Traditional ML/DL lifecycle | Large Language Model lifecycle |
| Data | Structured, images, audio | Unstructured text, embeddings |
| Tasks | CI/CD, drift monitoring | Prompt engineering, fine-tuning, RAG |
| Infra | GPUs optional | GPUs/TPUs heavy use |

---

### 11.3 Steps: Model Training vs Model Inference
**Model Training:**  
1. Data preprocessing  
2. Model selection  
3. Parameter & hyperparameter tuning  
4. Training  
5. Evaluation  
6. Registry storage  

**Model Inference:**  
1. Load model  
2. Preprocess input  
3. Predict  
4. Postprocess  
5. Return output  

---

### 11.4 Common Python Packages in ML
- **Data:** pandas, numpy  
- **Visualization:** matplotlib, seaborn, plotly  
- **ML:** scikit-learn, xgboost, lightgbm, catboost  
- **DL:** tensorflow, pytorch, keras  
- **Tracking:** mlflow  

---

### 11.5 What is an Outlier?
**A:** Data points that are too deviated from the usual data distribution. Anomalous records in datasets that may skew model results.  
**Detection:** Z-score, IQR, Isolation Forest.

---

### 11.6 Handling Null Values
**Steps:**  
1. Check % of missing data  
2. Impute (mean/median/mode, KNN)  
3. Drop nulls if low frequency and safe to remove  

---

### 11.7 How Big is Your Dataset?
**Categories:**  
- **Small (<1 GB):** pandas in-memory  - below 1 million samples - (no. of columns also matters)
- **Medium (1–100 GB):** Spark  - 1 to 10 million samples (no. of columns also matters)
- **Large (>100 GB):** Distributed cloud systems  - more than 10 million samples (no. of columns also matters)

---
