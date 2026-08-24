import pandas as pd
import matplotlib.pyplot as plt

# Create dataset
data = {
    "Name": [
        "Messi", "Ronaldo", "Neymar", "Mbappe", "Haaland",
        "Salah", "De Bruyne", "Kane", "Vinicius", "Bellingham"
    ],
    "Age": [39, 41, 34, 27, 26, 34, 35, 33, 26, 23],
    "Position": [
        "Forward", "Forward", "Forward", "Forward", "Forward",
        "Forward", "Midfielder", "Forward", "Forward", "Midfielder"
    ],
    "Goals": [30, 28, 25, 35, 32, 27, 12, 29, 24, 15],
    "Salary": [
        1200000, 1100000, 900000, 1000000, 950000,
        850000, 800000, 820000, 750000, 700000
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save CSV
df.to_csv("soccer_players.csv", index=False)

# Read CSV
df = pd.read_csv("soccer_players.csv")

print("Complete Dataset:")
print(df)

# Top 5 goals
print("\nTop 5 Players by Goals:")
print(df.nlargest(5, "Goals")[["Name", "Goals"]])

# Top 5 salaries
print("\nTop 5 Players by Salary:")
print(df.nlargest(5, "Salary")[["Name", "Salary"]])

# Average age
average_age = df["Age"].mean()

print("\nAverage Age:", round(average_age, 2))

# Players above average age
print("\nPlayers Above Average Age:")
print(df[df["Age"] > average_age][["Name", "Age"]])

# Position distribution
position_count = df["Position"].value_counts()

print("\nPosition Distribution:")
print(position_count)

# Bar chart
position_count.plot(kind="bar")

plt.xlabel("Position")
plt.ylabel("Number of Players")
plt.title("Distribution of Players by Position")
plt.xticks(rotation=0)
plt.show()
