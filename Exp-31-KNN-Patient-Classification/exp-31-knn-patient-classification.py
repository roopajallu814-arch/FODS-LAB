import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = {
    'Age': [25,30,35,40,45,50,55,60,28,38,48,58],
    'Gender': [0,1,0,1,0,1,0,1,1,0,1,0],
    'BP': [120,130,125,140,135,145,150,155,118,128,138,148],
    'Cholesterol': [180,190,175,220,210,230,240,250,170,200,225,245],
    'Outcome': ['Good','Good','Good','Bad','Good','Bad','Bad','Bad','Good','Good','Bad','Bad']
}
df = pd.DataFrame(data)
df['Outcome'] = df['Outcome'].map({'Bad': 0, 'Good': 1})

X = df[['Age','Gender','BP','Cholesterol']]
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Actual Values :", y_test.values)
print("Predicted Values :", y_pred)
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision :", precision_score(y_test, y_pred, zero_division=0))
print("Recall :", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score :", f1_score(y_test, y_pred, zero_division=0))
