import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
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

    # Self-contained dataset of Age and %Fat
    df = pd.DataFrame({
        "Age": [23, 23, 27, 27, 39, 41, 47, 49, 50, 52, 54, 54, 56, 57, 58, 58, 60, 61],
        "%Fat": [9.5, 26.5, 7.8, 17.8, 31.4, 25.9, 27.4, 27.2, 31.2, 34.6, 42.5, 28.8, 33.4, 30.2, 34.1, 32.9, 41.2, 35.7]
    })
    
    age = df["Age"]
    fat = df["%Fat"]
    
    # 1. Calculate Summary Statistics
    age_mean = age.mean()
    age_median = age.median()
    age_std = age.std()
    
    fat_mean = fat.mean()
    fat_median = fat.median()
    fat_std = fat.std()
    
    print("=== EXPERIMENT 21: STATISTICAL ANALYSIS OF AGE AND BODY FAT ===")
    print("\n--- Age Statistics ---")
    print(f"Mean:   {age_mean:.2f}")
    print(f"Median: {age_median:.2f}")
    print(f"Std Dev:{age_std:.2f}")
    
    print("\n--- %Fat Statistics ---")
    print(f"Mean:   {fat_mean:.2f}")
    print(f"Median: {fat_median:.2f}")
    print(f"Std Dev:{fat_std:.2f}")
    
    # 2. Boxplots of Age and %Fat
    plt.figure(figsize=(8, 5))
    plt.boxplot([age, fat], tick_labels=["Age", "%Fat"], patch_artist=True)
    plt.title("Boxplots of Age and %Fat")
    plt.ylabel("Value")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("output-1.png")
    plt.close()
    
    # 3. Scatter Plot between Age and %Fat
    plt.figure(figsize=(7, 5))
    plt.scatter(age, fat, color="blue", alpha=0.7, edgecolors="k")
    plt.title("Scatter Plot: Age vs %Fat")
    plt.xlabel("Age (years)")
    plt.ylabel("%Fat")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("output-2.png")
    plt.close()
    
    # 4. Q-Q Plots for Age and %Fat
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    stats.probplot(age, dist="norm", plot=axes[0])
    axes[0].set_title("Q-Q Plot for Age")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    
    stats.probplot(fat, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q Plot for %Fat")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("output-3.png")
    plt.close()
    
    print("\nSaved graphs as output-1.png (Boxplots), output-2.png (Scatter Plot), and output-3.png (Q-Q Plots).")

if __name__ == "__main__":
    main()
