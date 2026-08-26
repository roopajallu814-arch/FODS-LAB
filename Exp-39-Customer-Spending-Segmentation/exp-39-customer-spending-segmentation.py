import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = {
    'CustomerID': [1,1,2,2,3,3,4,4,5,5],
    'Amount': [100,150,200,250,50,70,500,600,800,900]
}
df = pd.DataFrame(data)

customer = df.groupby('CustomerID').agg(
    Spending=('Amount','sum'),
    Visits=('Amount','count')
).reset_index()

X = customer[['Spending','Visits']]

model = KMeans(n_clusters=3, random_state=42, n_init=10)
customer['Cluster'] = model.fit_predict(X)

print(customer)

plt.scatter(customer['Spending'], customer['Visits'], c=customer['Cluster'])
plt.xlabel("Total Spending")
plt.ylabel("Number of Visits")
plt.title("Customer Spending Segmentation")
plt.show()
