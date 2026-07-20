# List of item prices
prices = [50, 100, 30]

# List of quantities
quantities = [2, 1, 5]

# Discount and tax rates
discount_rate = 10
tax_rate = 5

# Calculate subtotal
subtotal = 0

for i in range(len(prices)):
    subtotal = subtotal + prices[i] * quantities[i]

# Calculate discount
discount = (subtotal * discount_rate) / 100

# Price after discount
discounted_price = subtotal - discount

# Calculate tax
tax = (discounted_price * tax_rate) / 100

# Final total
total = discounted_price + tax

# Display output
print("Subtotal =", subtotal)
print("Discount =", discount)
print("Price after Discount =", discounted_price)
print("Tax =", tax)
print("Total Cost =", total)
