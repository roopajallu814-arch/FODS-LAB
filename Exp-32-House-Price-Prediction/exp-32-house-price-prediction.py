import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    'Area': [800,1000,1200,1400,1600,1800,2000,2200],
    'Bedrooms': [2,2,3,3,3,4,4,4],
    'Age': [15,12,10,8,6,5,3,2],
    'Price': [40,50,60,70,80,90,105,115]
}
df = pd.DataFrame(data)

X = df[['Area','Bedrooms','Age']]
y = df['Price']

model = LinearRegression()
model.fit(X, y)

area = float(input("Enter house area: "))
bedrooms = int(input("Enter number of bedrooms: "))
age = int(input("Enter house age: "))

new_house = [[area, bedrooms, age]]
price = model.predict(new_house)

print("Predicted House Price:", round(price[0], 2), "Lakhs")
