import pandas as pd

# Sales Data
sales_data = {
    "Product": ["Laptop", "Mouse", "Keyboard", "Laptop", "Mouse",
                "Monitor", "Laptop", "Keyboard", "Monitor", "Mouse"],
    "Quantity": [5, 20, 10, 8, 15, 7, 12, 5, 6, 10]
}

# Create DataFrame
df = pd.DataFrame(sales_data)

# Find Top 5 Products
top_products = df.groupby("Product")["Quantity"].sum().sort_values(ascending=False).head(5)

print("Top 5 Products Sold:")
print(top_products)
