import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
import sys

class Tee:
    def __init__(self, filename):
        self.file = open(filename, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

def main():
    sys.stdout = Tee("output.txt")
    
    # Self-contained car dataset
    df = pd.DataFrame({
        "Age_Years": [1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 5, 7, 1, 4, 6],
        "Mileage_km": [15000, 25000, 35000, 45000, 60000, 75000, 90000, 110000, 20000, 30000, 55000, 85000, 12000, 50000, 70000],
        "Engine_Size_L": [2.0, 1.8, 2.0, 2.5, 1.6, 2.0, 1.8, 2.5, 2.5, 1.6, 2.0, 1.6, 2.5, 1.8, 2.5],
        "Price_USD": [28000, 24000, 21000, 20500, 15000, 13500, 11000, 9500, 26000, 18500, 17000, 11500, 31000, 17500, 14500]
    })
    
    features = ["Age_Years", "Mileage_km", "Engine_Size_L"]
    X = df[features]
    y = df["Price_USD"]
    
    # Initialize and train CART Regressor
    model = DecisionTreeRegressor(max_depth=3, random_state=42)
    model.fit(X, y)
    
    # Sample car features for prediction
    new_car = pd.DataFrame([{
        "Age_Years": 4,
        "Mileage_km": 45000,
        "Engine_Size_L": 2.0
    }])
    
    predicted_price = model.predict(new_car)[0]
    
    print("=== EXPERIMENT 28: CART FOR CAR PRICE PREDICTION ===")
    print("\n--- Model Training ---")
    print(f"Features Used: {', '.join(features)}")
    print(f"Target:        Price_USD")
    print(f"Tree Depth:    {model.get_depth()}")
    
    print("\n--- New Car Prediction ---")
    print(f"Input Features: Age = 4 years, Mileage = 45,000 km, Engine Size = 2.0 L")
    print(f"Predicted Price: ${predicted_price:,.2f}")
    
    # Decision Path Execution
    print("\n--- Decision Tree Rules (Text Format) ---")
    tree_text = export_text(model, feature_names=features)
    print(tree_text)
    
    print("\n--- Decision Path for New Car Sample ---")
    node_indicator = model.decision_path(new_car)
    leaf_id = model.apply(new_car)[0]
    node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
    
    for node_id in node_index:
        if leaf_id == node_id:
            print(f"-> Node {node_id} (LEAF): Predicted price = ${model.tree_.value[node_id][0][0]:,.2f}")
        else:
            feature_idx = model.tree_.feature[node_id]
            threshold = model.tree_.threshold[node_id]
            feature_name = features[feature_idx]
            val = new_car.iloc[0, feature_idx]
            comp = "<=" if val <= threshold else ">"
            print(f"-> Node {node_id}: {feature_name} ({val}) {comp} {threshold:.2f}")
            
    # Visualization: Decision Tree Diagram
    plt.figure(figsize=(12, 7))
    plot_tree(model, feature_names=features, filled=True, rounded=True, precision=2, fontsize=10)
    plt.title("CART Decision Tree Structure for Car Price Prediction")
    plt.tight_layout()
    plt.savefig("output.png")
    plt.close()
    
    print("\nDecision Tree structure saved to output.png.")

if __name__ == "__main__":
    main()
