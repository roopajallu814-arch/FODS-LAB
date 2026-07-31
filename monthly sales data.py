import matplotlib.pyplot as plt

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
sales = [120, 150, 180, 170, 200, 230]

plt.scatter(months, sales)

plt.title("Monthly Sales Scatter Plot")
plt.xlabel("Months")
plt.ylabel("Sales")

plt.show()
