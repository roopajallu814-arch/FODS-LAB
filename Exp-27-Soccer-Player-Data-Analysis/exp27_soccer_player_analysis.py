import pandas as pd
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
    
    # Self-contained player data
    df = pd.DataFrame({
        "Name": ["Lionel Messi", "Cristiano Ronaldo", "Kylian Mbappe", "Erling Haaland", "Kevin De Bruyne",
                 "Jude Bellingham", "Luka Modric", "Virgil van Dijk", "Ruben Dias", "Trent Alexander-Arnold",
                 "Alisson Becker", "Thibaut Courtois", "Pedri", "Bukayo Saka", "Achraf Hakimi"],
        "Position": ["Forward", "Forward", "Forward", "Forward", "Midfielder",
                     "Midfielder", "Midfielder", "Defender", "Defender", "Defender",
                     "Goalkeeper", "Goalkeeper", "Midfielder", "Forward", "Defender"],
        "Age": [36, 39, 25, 23, 32, 20, 38, 32, 26, 25, 31, 31, 21, 22, 25],
        "Goals": [28, 35, 30, 32, 10, 18, 5, 4, 2, 3, 0, 0, 6, 15, 4],
        "Salary_USD": [45000000, 50000000, 40000000, 35000000, 30000000,
                       25000000, 20000000, 22000000, 18000000, 16000000,
                       15000000, 17000000, 14000000, 18000000, 15000000]
    })
    
    print("=== EXPERIMENT 27: SOCCER PLAYER DATA ANALYSIS ===")
    
    # 1. Top 5 players by goals
    top5_goals = df.nlargest(5, "Goals")[["Name", "Position", "Goals"]]
    print("\n--- Top 5 Players by Goals ---")
    print(top5_goals.to_string(index=False))
    
    # 2. Top 5 players by salary
    top5_salary = df.nlargest(5, "Salary_USD")[["Name", "Position", "Salary_USD"]]
    print("\n--- Top 5 Players by Salary (USD) ---")
    print(top5_salary.to_string(index=False))
    
    # 3. Average player age
    avg_age = df["Age"].mean()
    print(f"\n--- Player Age Analysis ---")
    print(f"Average Player Age: {avg_age:.2f} years")
    
    # 4. Players above average age
    above_avg_age = df[df["Age"] > avg_age][["Name", "Age", "Position"]]
    print(f"\n--- Players Above Average Age (> {avg_age:.2f} years) ---")
    print(above_avg_age.to_string(index=False))
    
    # 5. Count players by position
    pos_counts = df["Position"].value_counts()
    print("\n--- Player Count by Position ---")
    print(pos_counts.to_string())
    
    # 6. Bar chart of position distribution
    plt.figure(figsize=(7, 5))
    bars = plt.bar(pos_counts.index, pos_counts.values, color=['#3498db', '#e74c3c', '#2ecc71', '#f1c40f'], edgecolor='black', alpha=0.85)
    plt.title("Soccer Player Distribution by Position")
    plt.xlabel("Position")
    plt.ylabel("Number of Players")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.1, f"{height}", 
                 ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig("output.png")
    plt.close()
    
    print("\nVisualization saved to output.png.")

if __name__ == "__main__":
    main()
