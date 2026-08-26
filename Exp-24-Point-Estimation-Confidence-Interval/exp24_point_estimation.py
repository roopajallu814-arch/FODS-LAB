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
    
    # Self-contained rare-element concentration dataset
    df = pd.DataFrame({
        "Sample_ID": list(range(1, 16)),
        "Concentration_ppm": [12.4, 14.1, 13.5, 12.8, 15.0, 13.2, 14.7, 12.9, 13.8, 14.3, 13.6, 14.0, 12.7, 13.9, 14.5]
    })
    concentrations = df["Concentration_ppm"]
    
    n = len(concentrations)
    
    # Point Estimate: Sample Mean
    point_estimate = concentrations.mean()
    sample_std = concentrations.std()
    
    # Confidence Level and Desired Precision Parameters
    confidence_level = 0.95
    desired_precision = 0.30  # Desired Margin of Error (E)
    
    # Confidence Interval calculation using t-distribution
    alpha = 1 - confidence_level
    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
    sem = sample_std / np.sqrt(n)
    margin_of_error = t_crit * sem
    
    ci_lower = point_estimate - margin_of_error
    ci_upper = point_estimate + margin_of_error
    
    # Required Sample Size for Desired Precision
    z_crit = stats.norm.ppf(1 - alpha/2)
    required_n = int(np.ceil((z_crit * sample_std / desired_precision)**2))
    
    print("=== EXPERIMENT 24: POINT ESTIMATION AND CONFIDENCE INTERVAL ===")
    print("\n--- Rare-Element Concentration Data Analysis ---")
    print(f"Sample Size (n):              {n}")
    print(f"Point Estimate (Sample Mean): {point_estimate:.4f} ppm")
    print(f"Sample Standard Deviation:    {sample_std:.4f} ppm")
    
    print(f"\n--- {confidence_level*100:.0f}% Confidence Interval ---")
    print(f"Standard Error (SEM):        {sem:.4f}")
    print(f"Margin of Error:             +/-{margin_of_error:.4f} ppm")
    print(f"Confidence Interval:         ({ci_lower:.4f}, {ci_upper:.4f}) ppm")
    
    print("\n--- Sample Size Estimation for Desired Precision ---")
    print(f"Target Margin of Error (E):  {desired_precision:.2f} ppm")
    print(f"Required Sample Size:        {required_n} samples")
    
    # Visualization
    plt.figure(figsize=(8, 5))
    plt.plot(df["Sample_ID"], concentrations, 'o-', color='#2c3e50', label='Sample Concentrations')
    plt.axhline(point_estimate, color='#e74c3c', linestyle='--', linewidth=2, label=f'Point Estimate (Mean = {point_estimate:.2f})')
    plt.axhspan(ci_lower, ci_upper, color='#e74c3c', alpha=0.2, label=f'{confidence_level*100:.0f}% Confidence Interval')
    
    plt.title("Rare-Element Concentration: Point Estimation & Confidence Interval")
    plt.xlabel("Sample ID")
    plt.ylabel("Concentration (ppm)")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("output.png")
    plt.close()
    
    print("\nVisualization saved to output.png.")

if __name__ == "__main__":
    main()
