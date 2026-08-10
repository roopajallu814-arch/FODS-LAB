import pandas as pd

# Temperature data
data = {
    "City": ["Chennai", "Chennai", "Chennai", "Chennai",
             "Delhi", "Delhi", "Delhi", "Delhi",
             "Bangalore", "Bangalore", "Bangalore", "Bangalore"],
    
    "Temperature": [32, 34, 31, 35,
                    25, 30, 20, 28,
                    25, 26, 24, 25]
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate mean temperature
mean_temp = df.groupby("City")["Temperature"].mean()

# Calculate standard deviation
std_temp = df.groupby("City")["Temperature"].std()

# Calculate temperature range
range_temp = df.groupby("City")["Temperature"].max() - \
             df.groupby("City")["Temperature"].min()

print("Mean Temperature:")
print(mean_temp)

print("\nStandard Deviation:")
print(std_temp)

print("\nTemperature Range:")
print(range_temp)

# Find highest temperature range
highest_range = range_temp.idxmax()

# Find most consistent city
most_consistent = std_temp.idxmin()

print("\nCity with Highest Temperature Range:", highest_range)
print("City with Most Consistent Temperature:", most_consistent)
