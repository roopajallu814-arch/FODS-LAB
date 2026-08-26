import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
    
    # Load Iris dataset directly from sklearn
    iris = load_iris(as_frame=True)
    df = iris.frame
    df['species'] = iris.target_names[iris.target]
    
    feature_cols = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
    X = df[feature_cols]
    y = df["species"]
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train Decision Tree Classifier
    clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("=== EXPERIMENT 29: DECISION TREE FOR IRIS FLOWER CLASSIFICATION ===")
    print("\n--- Model Training & Evaluation ---")
    print(f"Dataset Size:           {len(df)} samples")
    print(f"Features Used:          {', '.join(feature_cols)}")
    print(f"Target Species:         {list(np.unique(y))}")
    print(f"Test Set Accuracy:      {accuracy * 100:.2f}%")
    
    # New Flower Sample Prediction
    new_flower = pd.DataFrame([{
        "sepal length (cm)": 5.1,
        "sepal width (cm)": 3.5,
        "petal length (cm)": 1.4,
        "petal width (cm)": 0.2
    }])
    
    predicted_species = clf.predict(new_flower)[0]
    probabilities = clf.predict_proba(new_flower)[0]
    
    print("\n--- New Flower Sample Prediction ---")
    print(f"Input Measurements: Sepal Length = 5.1 cm, Sepal Width = 3.5 cm, Petal Length = 1.4 cm, Petal Width = 0.2 cm")
    print(f"PREDICTED SPECIES:  {predicted_species.upper()}")
    print("Class Probabilities:")
    for cls, prob in zip(clf.classes_, probabilities):
        print(f"  - {cls}: {prob:.2%}")
        
    # Decision Tree Rules Output
    print("\n--- Decision Tree Rules ---")
    tree_rules = export_text(clf, feature_names=feature_cols)
    print(tree_rules)
    
    # Visualization: Decision Tree Diagram
    plt.figure(figsize=(12, 8))
    plot_tree(clf, feature_names=feature_cols, class_names=clf.classes_, filled=True, rounded=True, fontsize=10)
    plt.title("Decision Tree Visualization for Iris Flower Classification")
    plt.tight_layout()
    plt.savefig("output.png")
    plt.close()
    
    print("\nDecision Tree structure saved to output.png.")

if __name__ == "__main__":
    main()
