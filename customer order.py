import pandas as pd

# Step 1: Create the DataFrame
order_data = pd.DataFrame({
    'Customer_ID': [101, 102, 101, 103, 102, 101],
    'Order_Date': ['2026-01-10', '2026-01-15', '2026-02-05',
                   '2026-02-10', '2026-03-01', '2026-03-15'],
    'Product_Name': ['Laptop', 'Mouse', 'Laptop',
                     'Keyboard', 'Mouse', 'Keyboard'],
    'Order_Quantity': [2, 5, 1, 3, 2, 4]
})

# Step 2: Display the DataFrame
print("Order Data:")
print(order_data)

# Step 3: Convert Order_Date to datetime format
order_data['Order_Date'] = pd.to_datetime(order_data['Order_Date'])

# Step 4: Total number of orders by each customer
orders_per_customer = order_data.groupby('Customer_ID').size()
print("\nTotal Orders by Each Customer:")
print(orders_per_customer)

# Step 5: Average order quantity for each product
average_quantity = order_data.groupby('Product_Name')['Order_Quantity'].mean()
print("\nAverage Order Quantity for Each Product:")
print(average_quantity)

# Step 6: Earliest order date
earliest_date = order_data['Order_Date'].min()
print("\nEarliest Order Date:", earliest_date.date())

# Step 7: Latest order date
latest_date = order_data['Order_Date'].max()
print("Latest Order Date:", latest_date.date())
