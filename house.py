import numpy as np

house_data = np.array([
    [3, 1500, 250000],
    [5, 2800, 450000],
    [4, 2200, 350000],
    [6, 3200, 550000],
    [5, 3000, 500000]
])

bedrooms = house_data[:, 0]
sale_price = house_data[:, 2]

average_price = np.mean(sale_price[bedrooms > 4])

print("Average Sale Price:", average_price)
