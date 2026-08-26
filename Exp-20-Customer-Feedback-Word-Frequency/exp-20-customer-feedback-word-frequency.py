import pandas as pd
import string
from collections import Counter
import matplotlib.pyplot as plt

# Read customer feedback data
df = pd.read_csv("data.csv")

# Combine all feedback
text = " ".join(df["feedback"])

# Convert to lowercase
text = text.lower()

# Remove punctuation
text = text.translate(str.maketrans("", "", string.punctuation))

# Stop words
stop_words = {
    "the", "and", "is", "a", "an", "of", "to",
    "for", "with", "in", "on", "this", "it"
}

# Split text into words
words = text.split()

# Remove stop words
words = [word for word in words if word not in stop_words]

# Calculate frequency
word_frequency = Counter(words)

# Get number of top words
n = int(input("Enter the number of top words: "))

top_words = word_frequency.most_common(n)

print("\nTop", n, "Most Frequent Words")
for word, count in top_words:
    print(word, ":", count)

# Prepare data for bar graph
words = [item[0] for item in top_words]
frequencies = [item[1] for item in top_words]

# Bar graph
plt.bar(words, frequencies)
plt.title("Top Most Frequent Words in Customer Feedback")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.show()
