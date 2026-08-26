import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'Engine': [1.2,1.5,1.8,2.0,2.2,2.5,3.0,3.5],
    'Horsepower': [80,100,120,140,160,180,220,250],
    'Mileage': [20,18,17,15,14,13,11,10],
    'Price': [6,8,10,13,16,19,25,30]
}
df = pd.DataFrame(data)

X = df[['Engine','Horsepower','Mileage']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Actual Prices :", y_test.values)
print("Predicted Prices :", y_pred)
print("\nMSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

print("\nFeature Importance:")
for feature, value in zip(X.columns, model.coef_):
    print(feature, "=", round(value, 3))
