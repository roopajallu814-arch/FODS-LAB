import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data = {
    'Spending': [100,120,150,500,550,600,1000,1100,1200],
    'Visits': [2,3,3,8,9,10,15,16,18],
    'Items': [3,4,5,10,12,11,20,22,25]
}
df = pd.DataFrame(data)

X = df[['Spending','Visits','Items']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(n_clusters=3, random_state=42, n_init=10)
model.fit(X_scaled)

spending = float(input("Enter spending: "))
visits = int(input("Enter visits: "))
items = int(input("Enter items purchased: "))

new_customer = [[spending, visits, items]]
new_scaled = scaler.transform(new_customer)
cluster = model.predict(new_scaled)

print("Customer belongs to Cluster:", cluster[0])
