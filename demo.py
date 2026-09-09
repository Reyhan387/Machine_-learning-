# Import libraries
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Create sample dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([3, 4, 2, 5, 6])

# Create Linear Regression model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict output
y_pred = model.predict(X)

# Display coefficients
print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])

# Calculate evaluation metrics
print("MAE:", mean_absolute_error(y, y_pred))
print("MSE:", mean_squared_error(y, y_pred))

rmse = np.sqrt(mean_squared_error(y, y_pred))
print("RMSE:", rmse)

print("R2 Score:", model.score(X, y))

# Visualize actual values
plt.scatter(X, y, label='Actual')

# Visualize regression line
plt.plot(X, y_pred, label='Regression Line')

plt.xlabel("X (Independent Variable)")
plt.ylabel("Y (Dependent Variable)")
plt.title("Linear Regression")
plt.legend()
plt.show()