orders = [
    {"customer": "Alice", "amount": 500},
    {"customer": "Bob", "amount": 750},
    {"customer": "Charlie", "amount": 1000}
]
total_orders = sum(order["amount"] for order in orders)
print("Orders:", orders)
print("Total:", total_orders)