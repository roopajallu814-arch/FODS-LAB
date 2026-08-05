import matplotlib.pyplot as plt

# Months
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Temperature Data
temperature = [22, 24, 28, 32, 35, 36, 34, 33, 31, 29, 25, 23]

# Line Plot
plt.plot(months, temperature, marker='o')
plt.title("Monthly Temperature")
plt.xlabel("Months")
plt.ylabel("Temperature (°C)")
plt.grid(True)
plt.show()

# Rainfall Data
rainfall = [15, 20, 35, 60, 110, 180, 220, 200, 170, 90, 40, 20]

# Scatter Plot
plt.scatter(months, rainfall)
plt.title("Monthly Rainfall")
plt.xlabel("Months")
plt.ylabel("Rainfall (mm)")
plt.grid(True)
plt.show()
