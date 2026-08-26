import pandas as pd
from sklearn.linear_model import LogisticRegression

data = {
    'Usage': [100,150,200,250,300,350,400,450,500,550],
    'Contract': [24,24,18,18,12,12,6,6,3,3],
    'Churn': [0,0,0,0,0,1,1,1,1,1]
}
df = pd.DataFrame(data)

X = df[['Usage','Contract']]
y = df['Churn']

model = LogisticRegression()
model.fit(X, y)

usage = float(input("Enter usage minutes: "))
contract = float(input("Enter contract duration: "))

prediction = model.predict([[usage, contract]])

if prediction[0] == 1:
    print("Prediction: Customer will CHURN")
else:
    print("Prediction: Customer will NOT CHURN")
