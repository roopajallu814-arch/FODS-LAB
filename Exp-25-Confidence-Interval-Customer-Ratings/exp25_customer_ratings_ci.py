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
    
    # Self-contained customer ratings dataset
    df = pd.DataFrame({
        "Review_ID": list(range(1, 31)),
        "Rating": [
            4.5, 4.0, 5.0, 3.5, 4.5, 5.0, 4.0, 3.0, 4.5, 5.0,
            4.0, 3.5, 4.5, 5.0, 4.0, 4.5, 3.0, 4.0, 5.0, 4.5,
            4.0, 3.5, 4.5, 5.0, 4.0, 4.5, 3.0, 4.0, 5.0, 4.5
        ]
    })
    ratings = df["Rating"]
    
    n = len(ratings)
    sample_mean = ratings.mean()
    sample_std = ratings.std()
    sem = stats.sem(ratings)
    
    # 95% Confidence Interval for True Population Mean Rating
    confidence_95 = 0.95
    t_crit_95 = stats.t.ppf((1 + confidence_95) / 2., n - 1)
    moe_95 = t_crit_95 * sem
    ci_95_lower = sample_mean - moe_95
    ci_95_upper = sample_mean + moe_95
    
    # 99% Confidence Interval for Comparison
    confidence_99 = 0.99
    t_crit_99 = stats.t.ppf((1 + confidence_99) / 2., n - 1)
    moe_99 = t_crit_99 * sem
    ci_99_lower = sample_mean - moe_99
    ci_99_upper = sample_mean + moe_99
    
    print("=== EXPERIMENT 25: CONFIDENCE INTERVAL FOR CUSTOMER RATINGS ===")
    print("\n--- Summary Statistics ---")
    print(f"Total Customer Reviews (n): {n}")
    print(f"Sample Mean Rating:        {sample_mean:.3f} / 5.0")
    print(f"Sample Std Deviation:      {sample_std:.3f}")
    print(f"Standard Error (SEM):      {sem:.3f}")
    
    print("\n--- 95% Confidence Interval for Population Mean ---")
    print(f"Margin of Error:           +/-{moe_95:.3f}")
    print(f"95% Confidence Interval:   ({ci_95_lower:.3f}, {ci_95_upper:.3f})")
    
    print("\n--- 99% Confidence Interval for Population Mean ---")
    print(f"Margin of Error:           +/-{moe_99:.3f}")
    print(f"99% Confidence Interval:   ({ci_99_lower:.3f}, {ci_99_upper:.3f})")
    
    # Visualization: Rating Distribution and Confidence Interval
    plt.figure(figsize=(8, 5))
    counts, bins, patches = plt.hist(ratings, bins=np.arange(2.75, 5.75, 0.5), 
                                     rwidth=0.8, color='#8e44ad', alpha=0.7, edgecolor='black')
    
    plt.axvline(sample_mean, color='#e74c3c', linestyle='--', linewidth=2.5, label=f'Sample Mean ({sample_mean:.2f})')
    plt.axvspan(ci_95_lower, ci_95_upper, color='#27ae60', alpha=0.3, label=f'95% CI ({ci_95_lower:.2f}, {ci_95_upper:.2f})')
    
    plt.title("Customer Ratings Distribution & 95% Confidence Interval")
    plt.xlabel("Rating (Stars)")
    plt.ylabel("Number of Reviews")
    plt.legend(loc="upper left")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("output.png")
    plt.close()
    
    print("\nVisualization saved to output.png.")

if __name__ == "__main__":
    main()
