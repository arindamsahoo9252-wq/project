import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
data = pd.read_csv("D:\\projrct 1\\project.csv")
print("Dataset Preview:")
print(data.head())
X = data[['Open', 'High', 'Low', 'Volume']]
y = data['Close']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error:", mse)
plt.figure()
plt.plot(y_test.values, label="Actual Price")
plt.plot(y_pred,label="Predicted Price")
plt.title("Stock Price Prediction")
plt.xlabel("Days")
plt.ylabel("Close Price")
plt.legend()
plt.show()
future_data = np.array([[160, 165, 158, 1300000]])
future_price = model.predict(future_data)
print("\nPredicted Future Closing Price:", future_price[0])