import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = {
    'Spending': [100,120,150,200,500,550,600,650,1000,1100],
    'Visits': [2,3,3,4,8,9,10,11,15,16],
    'Browsing': [5,6,7,8,15,16,17,18,25,26]
}
df = pd.DataFrame(data)

X = df[['Spending','Visits','Browsing']]

model = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = model.fit_predict(X)

print(df)

plt.scatter(df['Spending'], df['Visits'], c=df['Cluster'])
plt.xlabel("Spending")
plt.ylabel("Visits")
plt.title("Customer Segmentation using K-Means")
plt.show()
