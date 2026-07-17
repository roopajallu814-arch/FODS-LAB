import numpy as np

# Quarterly sales data
sales_data = np.array([50000, 60000, 70000, 80000])

# Calculate total sales
total_sales = np.sum(sales_data)

# Calculate percentage increase from Q1 to Q4
percentage_increase = ((sales_data[3] - sales_data[0]) / sales_data[0]) * 100

# Display results
print("Quarterly Sales:", sales_data)
print("Total Sales for the Year:", total_sales)
print("Percentage Increase from Q1 to Q4:", percentage_increase, "%")
