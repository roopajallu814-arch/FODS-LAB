import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
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
    
    # Self-contained patient dataset
    df = pd.DataFrame({
        "Patient_ID": list(range(1, 21)),
        "Age": [25, 30, 35, 28, 40, 45, 22, 38, 50, 33, 55, 60, 48, 52, 58, 62, 45, 50, 65, 56],
        "Glucose_Level": [85, 90, 95, 88, 100, 105, 82, 98, 110, 92, 160, 175, 150, 165, 180, 170, 155, 162, 185, 168],
        "Blood_Pressure": [70, 72, 74, 68, 76, 78, 66, 75, 80, 71, 88, 92, 85, 90, 94, 89, 86, 87, 95, 91],
        "Condition": ["Healthy"] * 10 + ["Diabetic"] * 10
    })
    
    feature_cols = ["Age", "Glucose_Level", "Blood_Pressure"]
    X = df[feature_cols]
    y = df["Condition"]
    
    # Standardize features for accurate distance computation in KNN
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Set number of neighbors K
    K = 3
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_scaled, y)
    
    # New patient input features
    new_patient_raw = pd.DataFrame([{
        "Age": 48,
        "Glucose_Level": 155,
        "Blood_Pressure": 86
    }])
    
    new_patient_scaled = scaler.transform(new_patient_raw)
    
    # Predict condition
    predicted_condition = knn.predict(new_patient_scaled)[0]
    class_probs = knn.predict_proba(new_patient_scaled)[0]
    
    # Find nearest neighbors
    distances, neighbor_indices = knn.kneighbors(new_patient_scaled)
    
    print("=== EXPERIMENT 30: K-NEAREST NEIGHBORS (KNN) PATIENT CLASSIFICATION ===")
    print(f"\n--- Model Setup ---")
    print(f"Dataset Size:         {len(df)} patients")
    print(f"Features Used:        {', '.join(feature_cols)}")
    print(f"Value of K (Neighbors): {K}")
    
    print("\n--- New Patient Feature Values ---")
    print(f"Age:            {new_patient_raw.iloc[0]['Age']} years")
    print(f"Glucose Level:  {new_patient_raw.iloc[0]['Glucose_Level']} mg/dL")
    print(f"Blood Pressure: {new_patient_raw.iloc[0]['Blood_Pressure']} mmHg")
    
    print(f"\n--- {K} Nearest Neighbors Information ---")
    for idx, (dist, neighbor_idx) in enumerate(zip(distances[0], neighbor_indices[0]), start=1):
        neighbor_data = df.iloc[neighbor_idx]
        print(f"  Neighbor {idx}: Patient ID #{neighbor_data['Patient_ID']} | Condition: {neighbor_data['Condition']} | Scaled Distance: {dist:.4f}")
        
    print("\n--- Prediction Output (Majority Voting) ---")
    print(f"PREDICTED CONDITION: {predicted_condition.upper()}")
    print("Class Probabilities:")
    for cls, prob in zip(knn.classes_, class_probs):
        print(f"  - {cls}: {prob:.2%}")
        
    # Visualization: 2D Scatter plot (Age vs Glucose Level)
    plt.figure(figsize=(8, 6))
    
    # Plot dataset points
    for condition, color in zip(["Healthy", "Diabetic"], ["#2ecc71", "#e74c3c"]):
        subset = df[df["Condition"] == condition]
        plt.scatter(subset["Age"], subset["Glucose_Level"], color=color, label=condition, s=80, alpha=0.8, edgecolors='k')
        
    # Plot new patient
    new_age = new_patient_raw.iloc[0]["Age"]
    new_glucose = new_patient_raw.iloc[0]["Glucose_Level"]
    plt.scatter(new_age, new_glucose, color="#3498db", marker="*", s=250, label="New Patient", edgecolors='black', zorder=5)
    
    # Draw lines to nearest neighbors
    for neighbor_idx in neighbor_indices[0]:
        n_age = df.iloc[neighbor_idx]["Age"]
        n_glucose = df.iloc[neighbor_idx]["Glucose_Level"]
        plt.plot([new_age, n_age], [new_glucose, n_glucose], 'k--', alpha=0.5)
        
    plt.title(f"KNN Classification (K={K}): Patient Condition Prediction")
    plt.xlabel("Age (years)")
    plt.ylabel("Glucose Level (mg/dL)")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("output.png")
    plt.close()
    
    print("\nVisualization saved to output.png.")

if __name__ == "__main__":
    main()
