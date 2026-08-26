import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
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

def calculate_ci(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    margin_of_error = sem * stats.t.ppf((1 + confidence) / 2., n - 1)
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error
    return mean, margin_of_error, ci_lower, ci_upper

def main():
    sys.stdout = Tee("output.txt")
    
    # Self-contained dataset
    df = pd.DataFrame({
        "Group": ["Drug"] * 20 + ["Placebo"] * 20,
        "BP_Reduction": [
            12.5, 14.2, 10.8, 15.1, 13.0, 11.5, 16.2, 14.0, 12.8, 13.5, 15.0, 11.2, 14.8, 13.9, 12.1, 15.5, 13.2, 14.1, 12.6, 13.7,
            3.1, 4.5, 2.8, 5.2, 1.9, 3.8, 4.0, 2.5, 3.4, 5.0, 2.1, 4.2, 3.9, 2.7, 4.8, 3.0, 2.6, 4.1, 3.3, 2.9
        ]
    })
    
    drug_data = df[df["Group"] == "Drug"]["BP_Reduction"]
    placebo_data = df[df["Group"] == "Placebo"]["BP_Reduction"]
    
    # Calculate CIs
    drug_mean, drug_moe, drug_lower, drug_upper = calculate_ci(drug_data, 0.95)
    placebo_mean, placebo_moe, placebo_lower, placebo_upper = calculate_ci(placebo_data, 0.95)
    
    print("=== EXPERIMENT 22: CONFIDENCE INTERVAL FOR BLOOD PRESSURE REDUCTION ===")
    print("\n--- Drug Group (95% CI) ---")
    print(f"Sample Size (n):      {len(drug_data)}")
    print(f"Mean Reduction:       {drug_mean:.2f} mmHg")
    print(f"Margin of Error:      +/-{drug_moe:.2f}")
    print(f"95% Confidence Interval: ({drug_lower:.2f}, {drug_upper:.2f}) mmHg")
    
    print("\n--- Placebo Group (95% CI) ---")
    print(f"Sample Size (n):      {len(placebo_data)}")
    print(f"Mean Reduction:       {placebo_mean:.2f} mmHg")
    print(f"Margin of Error:      +/-{placebo_moe:.2f}")
    print(f"95% Confidence Interval: ({placebo_lower:.2f}, {placebo_upper:.2f}) mmHg")
    
    # Visualization: Bar Chart with Error Bars (CI)
    groups = ["Drug", "Placebo"]
    means = [drug_mean, placebo_mean]
    errors = [drug_moe, placebo_moe]
    
    plt.figure(figsize=(7, 5))
    bars = plt.bar(groups, means, yerr=errors, capsize=7, color=['#2b5c8f', '#d95f02'], alpha=0.85)
    plt.title("Mean Blood Pressure Reduction with 95% Confidence Interval")
    plt.ylabel("Mean BP Reduction (mmHg)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar, mean, moe in zip(bars, means, errors):
        plt.text(bar.get_x() + bar.get_width()/2, mean / 2, f"{mean:.2f}\n(+/-{moe:.2f})", 
                 ha='center', va='center', color='white', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig("output.png")
    plt.close()
    
    print("\nVisualization saved to output.png.")

if __name__ == "__main__":
    main()
