import pandas as pd

# Likes received by posts
data = {
    "Likes": [10, 20, 10, 30, 20, 10, 40, 30, 20, 10, 50, 40, 30, 20, 10]
}

df = pd.DataFrame(data)

# Calculate frequency distribution
likes_frequency = df["Likes"].value_counts().sort_index()

print("Likes Data")
print(df)

print("\nFrequency Distribution of Likes")
print(likes_frequency)
