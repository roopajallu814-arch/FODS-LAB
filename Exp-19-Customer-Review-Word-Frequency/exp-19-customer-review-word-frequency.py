import pandas as pd
from collections import Counter

# Customer reviews dataset
data = {
    "Review": [
        "Good product and good quality",
        "Excellent product with good quality",
        "Good service and excellent product",
        "Product quality is good"
    ]
}

df = pd.DataFrame(data)

# Combine all reviews
text = " ".join(df["Review"])

# Convert text to lowercase
text = text.lower()

# Split into words
words = text.split()

# Calculate word frequency
word_frequency = Counter(words)

print("Customer Reviews")
print(df)

print("\nWord Frequency Distribution")
for word, count in word_frequency.most_common():
    print(word, ":", count)
