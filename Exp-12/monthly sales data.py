months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
sales = [1000, 1500, 1200, 1800, 2000, 2200, 2500, 2300, 2100, 2400, 2600, 2800]
print("Monthly sales data:")
for month, sale in zip(months, sales):
    print(f"{month}: {sale}")