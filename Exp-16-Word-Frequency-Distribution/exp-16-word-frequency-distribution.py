from collections import Counter

# Open and read the text file
file = open("sample_text.txt", "r")
text = file.read()
file.close()

# Convert text to lowercase
text = text.lower()

# Split text into words
words = text.split()

# Calculate word frequency
word_frequency = Counter(words)

print("Word Frequency Distribution")
print(word_frequency)
