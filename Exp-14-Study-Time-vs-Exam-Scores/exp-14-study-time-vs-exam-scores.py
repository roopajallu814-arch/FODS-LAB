import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Study_Time": [1, 2, 3, 4, 5, 6, 7, 8],
    "Exam_Score": [52, 58, 65, 72, 78, 85, 89, 94]
}

df = pd.DataFrame(data)

correlation = df["Study_Time"].corr(df["Exam_Score"])

print("Study Time and Exam Score Data")
print(df)

print("\nCorrelation Coefficient:", correlation)

plt.scatter(df["Study_Time"], df["Exam_Score"])
plt.title("Study Time vs Exam Score")
plt.xlabel("Study Time (Hours)")
plt.ylabel("Exam Score")
plt.grid(True)
plt.show()

