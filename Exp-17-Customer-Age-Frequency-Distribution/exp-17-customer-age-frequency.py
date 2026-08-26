import pandas as pd

# Customer age data
data = {
    "Age": [18, 20, 21, 22, 22, 22, 22, 25, 25, 28, 28, 28, 30]
}

df = pd.DataFrame(data)

# Calculate frequency distribution
age_frequency = df["Age"].value_counts().sort_index()

print("Customer Age Data")
print(df)

print("\nFrequency Distribution of Customer Ages")
print(age_frequency)
