# ============================================================
# BANK CUSTOMER SUBSCRIPTION PREDICTION
# AND CUSTOMER SEGMENTATION SYSTEM
# DSA0402 - FUNDAMENTALS OF DATA SCIENCE
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# ============================================================
# 1. LOAD DATASET
# ============================================================

FILE = "bank-full.csv"

if not os.path.exists(FILE):
    print("ERROR: bank-full.csv not found.")
    print("Place bank-full.csv in the same folder as this program.")
    exit()

df = pd.read_csv(FILE, sep=";")

print("\n" + "=" * 70)
print("BANK CUSTOMER SUBSCRIPTION PREDICTION SYSTEM")
print("=" * 70)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())


# ============================================================
# 2. BASIC DATA ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

target_counts = df["y"].value_counts()

print(target_counts)

print("\nTarget Percentage:")
print(df["y"].value_counts(normalize=True) * 100)


# ============================================================
# 3. PANDAS FILTERING
# ============================================================

print("\n" + "=" * 70)
print("PANDAS FILTERING OPERATIONS")
print("=" * 70)

high_balance = df[df["balance"] > 5000]

print("\nCustomers with balance > 5000:")
print(high_balance.head())

print("\nNumber of high-balance customers:")
print(len(high_balance))


# ============================================================
# 4. SORTING
# ============================================================

print("\nTop 10 Customers by Account Balance:")

sorted_balance = df.sort_values(
    by="balance",
    ascending=False
)

print(
    sorted_balance[
        ["age", "job", "balance", "housing", "loan", "y"]
    ].head(10)
)


# ============================================================
# 5. GROUPING AND AGGREGATION
# ============================================================

print("\n" + "=" * 70)
print("GROUPING BY JOB")
print("=" * 70)

job_analysis = df.groupby("job").agg(
    Total_Customers=("y", "count"),
    Subscribers=("y", lambda x: (x == "yes").sum()),
    Average_Balance=("balance", "mean")
)

job_analysis["Subscription_Rate"] = (
    job_analysis["Subscribers"] /
    job_analysis["Total_Customers"] * 100
)

print(job_analysis.sort_values(
    "Subscription_Rate",
    ascending=False
))


print("\n" + "=" * 70)
print("GROUPING BY EDUCATION")
print("=" * 70)

education_analysis = df.groupby("education").agg(
    Total_Customers=("y", "count"),
    Subscribers=("y", lambda x: (x == "yes").sum()),
    Average_Balance=("balance", "mean")
)

education_analysis["Subscription_Rate"] = (
    education_analysis["Subscribers"] /
    education_analysis["Total_Customers"] * 100
)

print(education_analysis)


# ============================================================
# 6. PREVIOUS CAMPAIGN OUTCOME ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("PREVIOUS CAMPAIGN OUTCOME ANALYSIS")
print("=" * 70)

poutcome_analysis = df.groupby("poutcome").agg(
    Total_Count=("y", "count"),
    Subscribed_Count=("y", lambda x: (x == "yes").sum()),
    Mean_Duration=("duration", "mean")
)

poutcome_analysis["Subscription_Rate_Pct"] = (
    poutcome_analysis["Subscribed_Count"] /
    poutcome_analysis["Total_Count"] * 100
)

print(poutcome_analysis)


# ============================================================
# 7. DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)

numeric_columns = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous"
]

statistics_table = pd.DataFrame(index=numeric_columns)

statistics_table["Mean"] = df[numeric_columns].mean()
statistics_table["Median"] = df[numeric_columns].median()
statistics_table["Variance"] = df[numeric_columns].var()
statistics_table["StdDev"] = df[numeric_columns].std()
statistics_table["IQR"] = (
    df[numeric_columns].quantile(0.75) -
    df[numeric_columns].quantile(0.25)
)
statistics_table["Minimum"] = df[numeric_columns].min()
statistics_table["Maximum"] = df[numeric_columns].max()
statistics_table["Skewness"] = df[numeric_columns].skew()

print(statistics_table)


# ============================================================
# 8. 95% CONFIDENCE INTERVAL
# ============================================================

print("\n" + "=" * 70)
print("STATISTICAL INFERENCE")
print("=" * 70)

# Mean balance
balance = df["balance"]

mean_balance = balance.mean()
std_balance = balance.std()
n_balance = len(balance)

standard_error = std_balance / np.sqrt(n_balance)

t_value = stats.t.ppf(
    0.975,
    df=n_balance - 1
)

margin_error = t_value * standard_error

balance_lower = mean_balance - margin_error
balance_upper = mean_balance + margin_error

print("\nPopulation Mean Customer Balance")
print("Mean:", mean_balance)
print("95% CI:", balance_lower, "to", balance_upper)


# Subscription rate

subscription_rate = (
    (df["y"] == "yes").sum() /
    len(df)
)

n = len(df)

se_rate = np.sqrt(
    subscription_rate *
    (1 - subscription_rate) / n
)

z = 1.96

rate_lower = subscription_rate - z * se_rate
rate_upper = subscription_rate + z * se_rate

print("\nPopulation Term Deposit Subscription Rate")
print("Rate:", subscription_rate)
print(
    "95% CI:",
    rate_lower,
    "to",
    rate_upper
)


# ============================================================
# 9. OUTLIER ANALYSIS USING IQR
# ============================================================

print("\n" + "=" * 70)
print("OUTLIER ANALYSIS")
print("=" * 70)

outlier_table = []

for column in numeric_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower) |
        (df[column] > upper)
    ]

    percentage = (
        len(outliers) / len(df) * 100
    )

    outlier_table.append([
        column,
        IQR,
        lower,
        upper,
        len(outliers),
        percentage
    ])

outlier_df = pd.DataFrame(
    outlier_table,
    columns=[
        "Attribute",
        "IQR",
        "Lower_Bound",
        "Upper_Bound",
        "Outlier_Count",
        "Outlier_Percentage"
    ]
)

print(outlier_df)


# ============================================================
# 10. EDA - HISTOGRAMS
# ============================================================

print("\nGenerating numerical histograms...")

for column in numeric_columns:

    plt.figure(figsize=(8, 5))

    plt.hist(
        df[column],
        bins=30,
        edgecolor="black"
    )

    plt.title(
        "Distribution of " + column
    )

    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        "histogram_" + column + ".png"
    )

    plt.show()


# ============================================================
# 11. EDA - BOX PLOTS
# ============================================================

for column in numeric_columns:

    plt.figure(figsize=(8, 4))

    plt.boxplot(df[column])

    plt.title(
        "Boxplot - " + column
    )

    plt.ylabel(column)

    plt.tight_layout()

    plt.savefig(
        "boxplot_" + column + ".png"
    )

    plt.show()


# ============================================================
# 12. CORRELATION MATRIX
# ============================================================

print("\nCorrelation Matrix:")

correlation_matrix = df[numeric_columns].corr()

print(correlation_matrix)

plt.figure(figsize=(10, 7))

plt.imshow(
    correlation_matrix,
    interpolation="nearest"
)

plt.colorbar()

plt.xticks(
    range(len(numeric_columns)),
    numeric_columns,
    rotation=45
)

plt.yticks(
    range(len(numeric_columns)),
    numeric_columns
)

plt.title("Correlation Matrix")

plt.tight_layout()

plt.savefig(
    "correlation_matrix.png"
)

plt.show()


# ============================================================
# 13. DATA PREPROCESSING
# ============================================================

print("\n" + "=" * 70)
print("DATA PREPROCESSING")
print("=" * 70)

# Convert binary variables

binary_columns = [
    "default",
    "housing",
    "loan"
]

for column in binary_columns:

    df[column] = df[column].map({
        "yes": 1,
        "no": 0
    })


# Target encoding

df["y"] = df["y"].map({
    "yes": 1,
    "no": 0
})


print("\nBinary encoding completed.")


# ============================================================
# 14. PRIMARY FEATURE SET
# ============================================================

# duration is deliberately excluded because it is known
# only after the call and therefore causes temporal leakage.

primary_features = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "campaign",
    "pdays",
    "previous",
    "poutcome"
]

X = df[primary_features]

y = df["y"]


# ============================================================
# 15. IDENTIFY CATEGORICAL AND NUMERICAL FEATURES
# ============================================================

categorical_features = [
    "job",
    "marital",
    "education",
    "contact",
    "month",
    "poutcome"
]

numerical_features = [
    "age",
    "default",
    "balance",
    "housing",
    "loan",
    "day",
    "campaign",
    "pdays",
    "previous"
]


# ============================================================
# 16. ONE-HOT ENCODING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ============================================================
# 17. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# ============================================================
# 18. MODEL CREATION
# ============================================================

models = {

    "kNN": Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=5
            )
        )
    ]),

    "Decision Tree": Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=8,
                class_weight="balanced",
                random_state=42
            )
        )
    ]),

    "Logistic Regression": Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            )
        )
    ])
}


# ============================================================
# 19. TRAIN AND EVALUATE MODELS
# ============================================================

results = []

predictions = {}

print("\n" + "=" * 70)
print("PRIMARY PRE-CONTACT MODEL EVALUATION")
print("=" * 70)

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(X_test)

    predictions[name] = y_pred

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    TN, FP, FN, TP = cm.ravel()

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1,
        TP,
        FP
    ])

    print("\nAccuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

    print("\nConfusion Matrix:")
    print(cm)


# ============================================================
# 20. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score",
        "True_Positives",
        "False_Positives"
    ]
)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 70)

print(results_df)


# ============================================================
# 21. PERFORMANCE GRAPH
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score"
]

for metric in metrics:

    plt.figure(figsize=(8, 5))

    plt.bar(
        results_df["Model"],
        results_df[metric]
    )

    plt.title(
        metric + " Comparison"
    )

    plt.ylabel(metric)

    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig(
        "model_" + metric.lower() + ".png"
    )

    plt.show()


# ============================================================
# 22. BEST MODEL
# ============================================================

best_index = results_df[
    "F1_Score"
].idxmax()

best_model_name = results_df.loc[
    best_index,
    "Model"
]

print("\nBest Deployable Model:")
print(best_model_name)


# ============================================================
# 23. K-MEANS CUSTOMER SEGMENTATION
# ============================================================

print("\n" + "=" * 70)
print("K-MEANS CUSTOMER SEGMENTATION")
print("=" * 70)

# Exclude target and duration

cluster_features = df[primary_features].copy()

# Convert categorical variables
cluster_encoded = pd.get_dummies(
    cluster_features,
    columns=categorical_features
)

# Convert boolean columns to integers
cluster_encoded = cluster_encoded.astype(float)

# Scaling
scaler = StandardScaler()

cluster_scaled = scaler.fit_transform(
    cluster_encoded
)


# ============================================================
# 24. ELBOW METHOD
# ============================================================

inertias = []

K_values = range(2, 9)

for k in K_values:

    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    km.fit(cluster_scaled)

    inertias.append(
        km.inertia_
    )


plt.figure(figsize=(8, 5))

plt.plot(
    K_values,
    inertias,
    marker="o"
)

plt.title(
    "Elbow Method for K-Means"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")

plt.tight_layout()

plt.savefig(
    "elbow_method.png"
)

plt.show()


# ============================================================
# 25. K-MEANS WITH K=4
# ============================================================

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(
    cluster_scaled
)

df["Cluster"] = clusters


# ============================================================
# 26. SILHOUETTE SCORE
# ============================================================

silhouette = silhouette_score(
    cluster_scaled,
    clusters
)

print("\nK = 4")
print("Inertia:", kmeans.inertia_)
print("Silhouette Score:", silhouette)


# ============================================================
# 27. PCA FOR VISUALIZATION
# ============================================================

pca = PCA(
    n_components=2,
    random_state=42
)

pca_result = pca.fit_transform(
    cluster_scaled
)

plt.figure(figsize=(9, 6))

plt.scatter(
    pca_result[:, 0],
    pca_result[:, 1],
    c=df["Cluster"],
    s=10
)

plt.title(
    "Customer Segmentation using K-Means and PCA"
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.tight_layout()

plt.savefig(
    "customer_clusters_pca.png"
)

plt.show()


# ============================================================
# 28. CLUSTER SUMMARY
# ============================================================

cluster_summary = df.groupby("Cluster").agg(
    Total_Customers=("y", "count"),
    Actual_Subscribers=("y", "sum"),
    Average_Age=("age", "mean"),
    Average_Balance=("balance", "mean"),
    Average_Campaign=("campaign", "mean")
)

cluster_summary[
    "Actual_Subscription_Rate_Pct"
] = (
    cluster_summary["Actual_Subscribers"] /
    cluster_summary["Total_Customers"] * 100
)

print("\n" + "=" * 70)
print("CUSTOMER CLUSTER SUMMARY")
print("=" * 70)

print(cluster_summary)


# ============================================================
# 29. ALIGN CLASSIFICATION WITH CLUSTERS
# ============================================================

best_predictions = predictions[
    best_model_name
]

# Re-index test predictions
test_prediction_df = X_test.copy()

test_prediction_df["Prediction"] = best_predictions

test_prediction_df.index = X_test.index

df["Predicted_Positive"] = 0

df.loc[
    test_prediction_df.index,
    "Predicted_Positive"
] = test_prediction_df["Prediction"]


# ============================================================
# 30. CLUSTER MARKETING PROFILE
# ============================================================

cluster_marketing = df.groupby("Cluster").agg(
    Total_Customers=("y", "count"),
    Actual_Subscribers=("y", "sum"),
    Predicted_Positives=("Predicted_Positive", "sum")
)

cluster_marketing[
    "Subscription_Rate_Pct"
] = (
    cluster_marketing["Actual_Subscribers"] /
    cluster_marketing["Total_Customers"] * 100
)

cluster_marketing[
    "Predicted_Positive_Rate_Pct"
] = (
    cluster_marketing["Predicted_Positives"] /
    cluster_marketing["Total_Customers"] * 100
)


print("\n" + "=" * 70)
print("CLASSIFICATION + CLUSTER ALIGNMENT")
print("=" * 70)

print(cluster_marketing)


# ============================================================
# 31. MARKETING RECOMMENDATIONS
# ============================================================

def recommendation(row):

    rate = row["Subscription_Rate_Pct"]

    if rate >= 18:
        return (
            "High Potential Segment - "
            "Prioritize high-touch telemarketing "
            "and digital re-engagement."
        )

    elif rate >= 12:
        return (
            "Moderate-High Potential Segment - "
            "Promote long-term savings and "
            "wealth preservation products."
        )

    elif rate >= 7:
        return (
            "Low Response Segment - "
            "Reduce intrusive telemarketing and "
            "use email/mobile campaigns."
        )

    else:
        return (
            "High Debt/Low Response Segment - "
            "Suppress aggressive deposit marketing "
            "and provide financial advisory support."
        )


cluster_marketing[
    "Strategic_Recommendation"
] = cluster_marketing.apply(
    recommendation,
    axis=1
)


print("\nMARKETING STRATEGY:")

for cluster, row in cluster_marketing.iterrows():

    print("\nCluster", cluster)
    print(
        "Customers:",
        row["Total_Customers"]
    )

    print(
        "Subscription Rate:",
        round(
            row["Subscription_Rate_Pct"],
            2
        ),
        "%"
    )

    print(
        "Recommendation:",
        row["Strategic_Recommendation"]
    )


# ============================================================
# 32. EXPORT PROCESSED DATA
# ============================================================

df.to_csv(
    "bank_processed.csv",
    index=False
)

cluster_summary.to_csv(
    "cluster_summary.csv"
)

cluster_marketing.to_csv(
    "marketing_strategy.csv"
)

results_df.to_csv(
    "model_performance.csv",
    index=False
)


# ============================================================
# 33. FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("FINAL PROJECT RESULT")
print("=" * 70)

print(
    "\nDataset Records:",
    len(df)
)

print(
    "Subscription Rate:",
    round(subscription_rate * 100, 2),
    "%"
)

print(
    "Best Deployable Model:",
    best_model_name
)

best_row = results_df[
    results_df["Model"] ==
    best_model_name
].iloc[0]

print(
    "Accuracy:",
    round(best_row["Accuracy"], 4)
)

print(
    "Precision:",
    round(best_row["Precision"], 4)
)

print(
    "Recall:",
    round(best_row["Recall"], 4)
)

print(
    "F1 Score:",
    round(best_row["F1_Score"], 4)
)

print(
    "K-Means Clusters:",
    4
)

print(
    "Silhouette Score:",
    round(silhouette, 4)
)

print("\nFiles generated:")
print("1. bank_processed.csv")
print("2. cluster_summary.csv")
print("3. marketing_strategy.csv")
print("4. model_performance.csv")
print("5. Histogram images")
print("6. Boxplot images")
print("7. Correlation matrix")
print("8. Elbow graph")
print("9. PCA cluster graph")
print("10. Model performance graphs")

print("\n" + "=" * 70)
print("IMPLEMENTATION COMPLETED SUCCESSFULLY")
print("=" * 70)
