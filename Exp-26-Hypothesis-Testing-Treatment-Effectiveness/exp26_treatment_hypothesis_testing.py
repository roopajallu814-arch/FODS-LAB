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

def main():
    sys.stdout = Tee("output.txt")
    
    # Self-contained dataset
    df = pd.DataFrame({
        "Patient_ID": list(range(1, 41)),
        "Group": ["Control"] * 20 + ["Treatment"] * 20,
        "Health_Score": [
            68.5, 72.0, 65.0, 70.5, 74.0, 66.5, 71.0, 69.5, 73.0, 67.0, 75.0, 68.0, 70.0, 72.5, 64.5, 71.5, 69.0, 73.5, 67.5, 70.0,
            82.0, 85.5, 78.0, 88.0, 84.5, 80.0, 86.0, 83.5, 87.0, 81.5, 89.0, 79.5, 85.0, 83.0, 86.5, 80.5, 88.5, 82.5, 87.5, 84.0
        ]
    })
    
    control_scores = df[df["Group"] == "Control"]["Health_Score"]
    treatment_scores = df[df["Group"] == "Treatment"]["Health_Score"]
    
    # Group Statistics
    c_mean, c_std, c_n = control_scores.mean(), control_scores.std(), len(control_scores)
    t_mean, t_std, t_n = treatment_scores.mean(), treatment_scores.std(), len(treatment_scores)
    
    # Perform Independent Two-Sample T-Test
    t_stat, p_val = stats.ttest_ind(treatment_scores, control_scores, equal_var=True)
    alpha = 0.05
    
    print("=== EXPERIMENT 26: HYPOTHESIS TESTING FOR TREATMENT EFFECTIVENESS ===")
    print("\n--- Group Summary Statistics ---")
    print(f"Control Group:   Mean = {c_mean:.2f}, Std = {c_std:.2f}, n = {c_n}")
    print(f"Treatment Group: Mean = {t_mean:.2f}, Std = {t_std:.2f}, n = {t_n}")
    
    print("\n--- Two-Sample T-Test Results ---")
    print(f"Test Statistic (t-score): {t_stat:.4f}")
    print(f"P-value:                  {p_val:.4e}")
    print(f"Significance Level (alpha):{alpha}")
    
    print("\n--- Conclusion ---")
    if p_val < alpha:
        print("Reject Null Hypothesis (H0).")
        print("Conclusion: The treatment is STATISTICALLY SIGNIFICANT and effective compared to the control group.")
    else:
        print("Fail to Reject Null Hypothesis (H0).")
        print("Conclusion: No statistically significant difference between treatment and control groups.")
        
    # Visualization: Boxplot Comparison
    plt.figure(figsize=(7, 5))
    plt.boxplot([control_scores, treatment_scores], tick_labels=["Control Group", "Treatment Group"], patch_artist=True,
                boxprops=dict(facecolor='#16a085', color='black'),
                medianprops=dict(color='yellow', linewidth=2))
    
    plt.title("Treatment Effectiveness: Health Scores Comparison")
    plt.ylabel("Health Score")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("output.png")
    plt.close()
    
    print("\nVisualization saved to output.png.")

if __name__ == "__main__":
    main()
