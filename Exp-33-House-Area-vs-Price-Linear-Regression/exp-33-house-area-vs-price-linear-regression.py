import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'Area': [800,1000,1200,1400,1600,1800,2000,2200],
    'Price': [40,50,60,70,80,90,105,115]
}
df = pd.DataFrame(data)

X = df[['Area']]
y = df['Price']

plt.scatter(df['Area'], df['Price'])
plt.xlabel("House Area")
plt.ylabel("House Price")
plt.title("House Area vs Price")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Actual Prices :", y_test.values)
print("Predicted Prices :", y_pred)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
