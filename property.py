import pandas as pd

# Create Property Data
data = {
    "Property_ID": [101, 102, 103, 104, 105],
    "Location": ["Chennai", "Hyderabad", "Chennai", "Bangalore", "Hyderabad"],
    "Bedrooms": [3, 5, 2, 6, 4],
    "Area_sqft": [1200, 2500, 1000, 3000, 1800],
    "Listing_Price": [5500000, 8500000, 4500000, 12000000, 7000000]
}

property_data = pd.DataFrame(data)

print("Property Data")
print(property_data)

# 1. Average listing price in each location
avg_price = property_data.groupby("Location")["Listing_Price"].mean()
print("\nAverage Listing Price by Location")
print(avg_price)

# 2. Number of properties with more than four bedrooms
count = property_data[property_data["Bedrooms"] > 4].shape[0]
print("\nProperties with more than four bedrooms:", count)

# 3. Property with the largest area
largest = property_data.loc[property_data["Area_sqft"].idxmax()]
print("\nProperty with Largest Area")
print(largest)
