# Regularized Models 
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Create synthetic regression data
X, y = make_regression(n_samples=100, n_features=20, noise=0.3, random_state=42)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)

# Elastic Net
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)  # l1_ratio controls mix of L1 and L2
elastic.fit(X_train, y_train)
elastic_pred = elastic.predict(X_test)

# Evaluate
print("Ridge MSE:", mean_squared_error(y_test, ridge_pred))
print("Lasso MSE:", mean_squared_error(y_test, lasso_pred))
print("Elastic Net MSE:", mean_squared_error(y_test, elastic_pred))

import matplotlib.pyplot as plt
x=['Ridge','Lasso','Elastic Net']
y=[mean_squared_error(y_test, ridge_pred),mean_squared_error(y_test, lasso_pred),mean_squared_error(y_test, elastic_pred)]

plt.bar(x,y,color=['red','blue','green'],width=0.2)
plt.show()






