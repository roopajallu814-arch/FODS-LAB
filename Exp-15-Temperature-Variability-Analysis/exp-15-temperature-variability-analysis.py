import pandas as pd

# Temperature data for different cities
data = {
    "City": [
        "Chennai", "Chennai", "Chennai", "Chennai", "Chennai",
        "Delhi", "Delhi", "Delhi", "Delhi", "Delhi",
        "Mumbai", "Mumbai", "Mumbai", "Mumbai", "Mumbai"
    ],
    "Temperature": [
        30, 32, 31, 33, 29,
        20, 25, 18, 30, 22,
        28, 30, 29, 31, 27
    ]
}

df = pd.DataFrame(data)

# Calculate mean temperature
mean_temperature = df.groupby("City")["Temperature"].mean()

# Calculate standard deviation
std_temperature = df.groupby("City")["Temperature"].std()

# Calculate temperature range
temperature_range = (
    df.groupby("City")["Temperature"].max()
    - df.groupby("City")["Temperature"].min()
)

print("Temperature Data")
print(df)

print("\nMean Temperature:")
print(mean_temperature)

print("\nStandard Deviation:")
print(std_temperature)

print("\nTemperature Range:")
print(temperature_range)

# Find city with highest temperature range
highest_range_city = temperature_range.idxmax()

# Find city with lowest standard deviation
most_consistent_city = std_temperature.idxmin()

print("\nCity with Highest Temperature Range:", highest_range_city)
print("Most Consistent City:", most_consistent_city)
