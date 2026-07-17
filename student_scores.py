import numpy as np

student_scores = np.array([
    [80, 75, 90, 85],
    [70, 80, 85, 90],
    [90, 85, 88, 92],
    [85, 78, 84, 88]
])

subjects = ["Math", "Science", "English", "History"]

average_scores = np.mean(student_scores, axis=0)

highest = np.argmax(average_scores)

print("Average Score of Each Subject")

for i in range(len(subjects)):
    print(subjects[i], ":", average_scores[i])

print("\nSubject with Highest Average Score:", subjects[highest])
print("Highest Average Score:", average_scores[highest])
