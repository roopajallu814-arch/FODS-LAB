import matplotlib.pyplot as plt

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
sales = [25000, 30000, 28000, 35000, 40000, 45000]

plt.plot(months, sales, marker='o', linewidth=2)

plt.title("Monthly Sales Line Plot")
plt.xlabel("Months")
plt.ylabel("Sales")
plt.grid(True)

plt.show()
