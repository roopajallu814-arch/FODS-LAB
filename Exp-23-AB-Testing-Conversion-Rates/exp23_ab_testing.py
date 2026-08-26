import numpy as np
import pandas as pd
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
    
    # Dataset parameters: Design A vs Design B
    visitors_A, conversions_A = 1000, 120
    visitors_B, conversions_B = 1000, 160
    
    rate_A = conversions_A / visitors_A
    rate_B = conversions_B / visitors_B
    
    print("=== EXPERIMENT 23: A/B TESTING OF WEBSITE CONVERSION RATES ===")
    print("\n--- Summary Data ---")
    print(f"Design A: {conversions_A} conversions / {visitors_A} visitors (Conversion Rate: {rate_A:.2%})")
    print(f"Design B: {conversions_B} conversions / {visitors_B} visitors (Conversion Rate: {rate_B:.2%})")
    
    # Perform Two-Proportion Z-Test
    # Pooled proportion
    p_pool = (conversions_A + conversions_B) / (visitors_A + visitors_B)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/visitors_A + 1/visitors_B))
    
    z_stat = (rate_B - rate_A) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    alpha = 0.05
    
    print("\n--- Statistical Test Results (Two-Proportion Z-Test) ---")
    print(f"Test Statistic (Z-score): {z_stat:.4f}")
    print(f"P-value:                  {p_value:.4f}")
    print(f"Significance Level (alpha):{alpha}")
    
    print("\n--- Conclusion ---")
    if p_value < alpha:
        print("Reject the Null Hypothesis (H0).")
        print("Conclusion: There is a statistically SIGNIFICANT difference in conversion rates between Design A and Design B.")
    else:
        print("Fail to Reject the Null Hypothesis (H0).")
        print("Conclusion: There is NO statistically significant difference in conversion rates between Design A and Design B.")
        
    # Visualization
    designs = ["Design A", "Design B"]
    rates = [rate_A * 100, rate_B * 100]
    
    plt.figure(figsize=(7, 5))
    bars = plt.bar(designs, rates, color=['#3498db', '#2ecc71'], alpha=0.85, width=0.5)
    plt.title("A/B Test: Website Conversion Rates Comparison")
    plt.ylabel("Conversion Rate (%)")
    plt.ylim(0, max(rates) + 5)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{height:.2f}%", 
                 ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig("output.png")
    plt.close()
    
    print("\nVisualization saved to output.png.")

if __name__ == "__main__":
    main()
