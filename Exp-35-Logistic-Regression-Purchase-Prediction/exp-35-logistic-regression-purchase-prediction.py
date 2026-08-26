import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = {
    'Age': [20,25,30,35,40,45,50,55,60,65],
    'Salary': [20,25,30,35,40,45,50,55,60,65],
    'Purchased': [0,0,0,1,1,1,1,1,1,1]
}
df = pd.DataFrame(data)

X = df[['Age','Salary']]
y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Actual Values :", y_test.values)
print("Predicted Values :", y_pred)
print("\nAccuracy :", accuracy_score(y_test, y_pred))
print("Precision :", precision_score(y_test, y_pred, zero_division=0))
print("Recall :", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score :", f1_score(y_test, y_pred, zero_division=0))
