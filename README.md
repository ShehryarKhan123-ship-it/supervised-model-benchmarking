# 🧠 Classifier Model Evaluation on Multiclass Diabetes Dataset

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Model Accuracy](https://img.shields.io/badge/LogReg%20Accuracy-89%25-brightgreen)
![MSE](https://img.shields.io/badge/Best%20MSE-0.21-lightgrey)

A comprehensive evaluation and comparison of machine learning classifiers—including Logistic Regression, SGDClassifier, and GridSearchCV-tuned SGDClassifier—on a multiclass, uncleaned diabetes dataset. The objective is to explore model performance, analyze hyperparameter impacts, and investigate the effects of complexity (e.g., regularization and loss functions) using robust evaluation metrics like MSE and RMSE.

---

## 📊 Dataset Overview

- **Type**: Multiclass Classification
- **Domain**: Medical / Health (Diabetes)
- **Format**: Raw & uncleaned dataset
- **Features**: Various clinical metrics (glucose, insulin, BMI, etc.)
- **Target Classes**: Multiple diabetes diagnosis categories

---

## 🔄 Project Pipeline

### 1. **Data Preprocessing**
- Handled missing/null values
- Converted categorical features using encoding techniques (LabelEncoder, OneHotEncoder where needed)
- Normalized/standardized numerical features using `StandardScaler`
- Split dataset into training and testing subsets (e.g., 80/20 ratio)

### 2. **Model Training**

#### ✅ Logistic Regression
- Baseline linear classifier for multiclass prediction
- Regularization: `L2`
- Solver: `lbfgs`

#### ✅ SGDClassifier
- Used for faster training on large datasets
- Parameters tuned manually for loss = `log_loss`, `l1/l2/elasticnet` penalties

#### ✅ GridSearchCV with SGDClassifier
- Exhaustive search over the following parameter grid:
  - `alpha`: [0.0001, 0.001, 0.01, 0.1, 1]
  - `penalty`: ['l2', 'l1', 'elasticnet']
  - `loss`: ['log_loss']
  - `max_iter`: [1000]
  - `tol`: [1e-3]
- Used 5-fold Stratified Cross-Validation

---

## 📈 Evaluation Metrics

All models were evaluated using the following metrics:

- **MSE (Mean Squared Error)**
- **RMSE (Root Mean Squared Error)**
- **Cross-Validation Accuracy**
- **Best Parameters (for GridSearchCV)**

---

## 📌 Results

| Model                     | MSE   | RMSE  | Cross-Validation Accuracy |
|--------------------------|-------|-------|----------------------------|
| Logistic Regression      | 0.210 | 0.458 | ~89%                       |
| SGDClassifier            | 0.450 | 0.671 | ~36%                       |
| GridSearchCV-SGDClassifier | 0.453 | 0.672 | ~36%                       |

- ✅ **Logistic Regression** achieved the best generalization and lowest error.
- ⚠️ **SGDClassifier** underperformed due to limited hyperparameter tuning.
- ⚙️ **GridSearchCV-SGDClassifier** did not yield improvement—suggesting overfitting or unsuitable grid range.

---
## 📚 Learnings & Insights

- Logistic Regression generalizes better due to simpler hypothesis space.
- SGDClassifier can underperform if hyperparameters aren’t well-tuned.
- GridSearchCV may not improve performance if the parameter space is misaligned.
- Overfitting is likely with high model complexity or poorly regularized training.

---

## 🚀 Practical Applications

- **Model 1** (Logistic Regression) is ideal for clinical diagnostics or deployment.
- **Other models** offer value for experimentation, tuning studies, or instructional purposes.

---
## 🧾 Report

A well-designed professional report summarizing methodology and analysis:

📄 [`Model_Evaluation_Report.pdf`](./reports/Model_Evaluation_Report.pdf)

---


## 📚 Key Learnings

- **Model Complexity**: Logistic Regression, although simpler, generalized better than more flexible SGD-based classifiers.
- **Overfitting**: Tuning with GridSearchCV did not always improve performance; excessive flexibility (elasticnet, low alpha) may harm generalization.
- **Polynomial Degrees (if applicable)**: Higher degrees lead to overfitting unless regularized effectively.

---

## 🚀 Practical Implications

- **Healthcare Tools**: Model 1 (Logistic Regression) is suitable for integration in diagnostic decision systems due to consistent and low error.
- **Educational Use**: Other models are ideal for studying hyperparameter sensitivity and evaluation methodology.

---

## 🧾 Report

A detailed PDF report containing methodology, results, analysis, and recommendations is included in this repository:

📄 `Model_Evaluation_Report.pdf`

---

## 📦 Tech Stack

- Python
- scikit-learn
- pandas / numpy
- matplotlib / seaborn (for optional visualization)
- FPDF (for report generation)

---

## 📁 Repository Structure
├── data/
│ └── diabetes.csv
├── notebooks/
│ └── model_training.ipynb
├── reports/
│ └── Model_Evaluation_Report.pdf
├── src/
│ └── preprocess.py
│ └── train.py
├── README.md


---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests for improvements or new evaluation techniques.

---



## 🔧 Setup & Installation

```bash
# Clone the repo
git clone https://github.com/your-username/classifier-model-evaluation.git
cd classifier-model-evaluation

# Create virtual environment and activate
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```
---

## 📬 Contact

For queries or collaborations:  
**Your Name** – [Shehryar Khan]    
GitHub: [https://github.com/ShehryarKhan123-ship-it/supervised-model-benchmarking]

---

