import numpy as np

fuel_efficiency = np.array([20, 25, 30, 35])

average = np.mean(fuel_efficiency)

old_model = fuel_efficiency[0]
new_model = fuel_efficiency[3]

improvement = ((new_model - old_model) / old_model) * 100

print("Fuel Efficiency (MPG):", fuel_efficiency)
print("Average Fuel Efficiency:", average)
print("Old Model Fuel Efficiency:", old_model)
print("New Model Fuel Efficiency:", new_model)
print("Percentage Improvement:", improvement)
