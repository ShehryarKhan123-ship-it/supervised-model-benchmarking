import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, mean_squared_error

# -----------------------------------
# 1. Load and preprocess data
# -----------------------------------
df = pd.read_csv("C:/Users/ART/Desktop/diabetes_unclean.csv")

# Fill missing values
df['AGE'] = df['AGE'].fillna(df['AGE'].mode()[0])
df.dropna(subset=['Urea', 'Cr', 'HbA1c'], inplace=True)

for col in ['HDL', 'LDL', 'VLDL', 'TG', 'Chol']:
    df[col] = df[col].fillna(df[col].mean())

print("Missing values after cleaning:")
print(df.isnull().sum())

# -----------------------------------
# 2. Scale numerical columns
# -----------------------------------
scaler = StandardScaler()
num_cols = ['Urea', 'Cr', 'HbA1c', 'Chol', 'HDL', 'LDL', 'VLDL', 'TG']

df[num_cols] = scaler.fit_transform(df[num_cols])

# -----------------------------------
# 3. Encode categorical variables
# -----------------------------------
gender_map = {'M': 0, 'F': 1}
class_map = {'N': 0, 'P': 1, 'Y': 2}

df['Gender'] = df['Gender'].str.upper().map(gender_map)
df['CLASS'] = df['CLASS'].map(class_map)

df.dropna(subset=['CLASS'], inplace=True)

print(df.info())

# -----------------------------------
# 4. Prepare features and target
# -----------------------------------
X = df.drop(columns=['CLASS', 'Gender'])
y = df['CLASS']

# Stratified split to keep class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------------
# 5. Train Logistic Regression model
# -----------------------------------
print("Training Logistic Regression...")
start_time = time.time()
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
end_time = time.time()

lr_train_time = end_time - start_time
y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_lr))
print("Logistic Regression ROC-AUC (OvR):", roc_auc_score(y_test, y_prob_lr, multi_class='ovr'))
print("Logistic Regression MSE:", mean_squared_error(y_test, y_pred_lr))

cm_lr = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(6,4))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues')
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# -----------------------------------
# 6. Train SGDClassifier (default)
# -----------------------------------
print("Training SGDClassifier (default)...")
start_time = time.time()
sgd = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd.fit(X_train, y_train)
end_time = time.time()

sgd_train_time = end_time - start_time
y_pred_sgd = sgd.predict(X_test)

print("SGDClassifier Accuracy:", accuracy_score(y_test, y_pred_sgd))
print("SGDClassifier Classification Report:\n", classification_report(y_test, y_pred_sgd))
print("SGDClassifier MSE:", mean_squared_error(y_test, y_pred_sgd))

cm_sgd = confusion_matrix(y_test, y_pred_sgd)
plt.figure(figsize=(6,4))
sns.heatmap(cm_sgd, annot=True, fmt='d', cmap='Greens')
plt.title("SGDClassifier Confusion Matrix")
plt.show()

# -----------------------------------
# 7. Train SGDClassifier with GridSearchCV
# -----------------------------------
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import time

# Parameter grid with more flexible loss functions
params = {
    'alpha': [0.0001, 0.001, 0.01],
    'penalty': ['l2', 'l1'],  # dropped 'elasticnet' for now, can add back later
    'loss': ['hinge', 'log_loss', 'modified_huber'],
    'max_iter': [1000],
    'tol': [1e-3]
}

print("Training SGDClassifier with GridSearchCV...")
start_time = time.time()
grid = GridSearchCV(
    estimator=SGDClassifier(random_state=42),
    param_grid=params,
    scoring='accuracy',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1
)
grid.fit(X_train, y_train)
end_time = time.time()

grid_train_time = end_time - start_time
best_sgd = grid.best_estimator_

print("✅ Best Parameters for SGDClassifier:", grid.best_params_)
print("✅ Best Cross-Validation Accuracy:", round(grid.best_score_, 4))


y_pred_grid = best_sgd.predict(X_test)

print("SGDClassifier (GridSearchCV) Accuracy:", accuracy_score(y_test, y_pred_grid))
print("SGDClassifier (GridSearchCV) Classification Report:\n", classification_report(y_test, y_pred_grid))
print("SGDClassifier (GridSearchCV) MSE:", mean_squared_error(y_test, y_pred_grid))

cm_grid = confusion_matrix(y_test, y_pred_grid)
plt.figure(figsize=(6,4))
sns.heatmap(cm_grid, annot=True, fmt='d', cmap='Oranges')
plt.title("SGDClassifier GridSearchCV Confusion Matrix")
plt.show()

# -----------------------------------
# 8. Plot Training Time and Accuracy Comparison
# -----------------------------------
models = ['Logistic Regression', 'SGDClassifier Default', 'SGDClassifier GridSearchCV']
train_times = [lr_train_time, sgd_train_time, grid_train_time]
accuracies = [
    accuracy_score(y_test, y_pred_lr),
    accuracy_score(y_test, y_pred_sgd),
    accuracy_score(y_test, y_pred_grid)
]

fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Training Time plot
axs[0].bar(models, train_times, color=['blue', 'green', 'orange'])
axs[0].set_title('Training Time Comparison (seconds)')
axs[0].set_ylabel('Time (s)')

# Accuracy plot
axs[1].bar(models, accuracies, color=['blue', 'green', 'orange'])
axs[1].set_title('Accuracy Comparison')
axs[1].set_ylabel('Accuracy')
axs[1].set_ylim(0, 1)

plt.tight_layout()
plt.show()

# -----------------------------------
# 9. Calculate ROC-AUC (OvR) and MSE for all models
# -----------------------------------
from sklearn.preprocessing import label_binarize

# Binarize y_test for multiclass ROC-AUC
classes = np.unique(y)
y_test_bin = label_binarize(y_test, classes=classes)

# Logistic Regression ROC-AUC
roc_auc_lr = roc_auc_score(y_test_bin, lr.predict_proba(X_test), multi_class='ovr')
mse_lr = mean_squared_error(y_test, y_pred_lr)

# SGDClassifier default ROC-AUC (handle no predict_proba)
if hasattr(sgd, "predict_proba"):
    probs_sgd = sgd.predict_proba(X_test)
else:
    # Use decision_function + softmax approximation
    decision_scores = sgd.decision_function(X_test)
    exp_scores = np.exp(decision_scores - np.max(decision_scores, axis=1, keepdims=True))
    probs_sgd = exp_scores / exp_scores.sum(axis=1, keepdims=True)

roc_auc_sgd = roc_auc_score(y_test_bin, probs_sgd, multi_class='ovr')
mse_sgd = mean_squared_error(y_test, y_pred_sgd)

# SGDClassifier GridSearchCV ROC-AUC (handle no predict_proba)
if hasattr(best_sgd, "predict_proba"):
    probs_grid = best_sgd.predict_proba(X_test)
else:
    decision_scores = best_sgd.decision_function(X_test)
    exp_scores = np.exp(decision_scores - np.max(decision_scores, axis=1, keepdims=True))
    probs_grid = exp_scores / exp_scores.sum(axis=1, keepdims=True)

roc_auc_grid = roc_auc_score(y_test_bin, probs_grid, multi_class='ovr')
mse_grid = mean_squared_error(y_test, y_pred_grid)

# -----------------------------------
# 10. Plot ROC-AUC Curves and MSE Comparison
# -----------------------------------
from sklearn.metrics import roc_curve, auc

plt.figure(figsize=(12, 8))

colors = ['blue', 'green', 'orange']
models = ['Logistic Regression', 'SGDClassifier Default', 'SGDClassifier GridSearchCV']
probs_list = [lr.predict_proba(X_test), probs_sgd, probs_grid]

for i, (model_name, probs) in enumerate(zip(models, probs_list)):
    for class_idx, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], probs[:, class_idx])
        roc_auc = auc(fpr, tpr)
        linestyle = '-' if i == 0 else '--' if i == 1 else '-.'
        plt.plot(fpr, tpr, color=colors[i], linestyle=linestyle,
                 label=f'{model_name} Class {cls} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.title('ROC Curves (One-vs-Rest) for Multiclass Classification')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right', fontsize=8)
plt.grid(True)
plt.show()

# Mean Squared Error bar plot
mses = [mse_lr, mse_sgd, mse_grid]

plt.figure(figsize=(8,5))
plt.bar(models, mses, color=colors)
plt.title('Mean Squared Error (MSE) Comparison')
plt.ylabel('MSE')
plt.ylim(0, max(mses)*1.2)
for i, v in enumerate(mses):
    plt.text(i, v + max(mses)*0.05, f"{v:.4f}", ha='center')
plt.show()
