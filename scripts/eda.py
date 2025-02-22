# scripts/eda.py

import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df):
    print("\n----- DATA OVERVIEW -----")
    print("Shape:", df.shape)
    print("\nHead:\n", df.head())
    print("\nData Types:\n", df.dtypes)
    print("\nSummary Statistics:\n", df.describe())

    missing = df.isnull().sum()
    print("\nMissing Values:\n", missing[missing > 0])

    # Example: Plot distribution of 'Fraud_Type'
    if 'Fraud_Type' in df.columns:
        plt.figure(figsize=(8,4))
        order = df['Fraud_Type'].value_counts().index
        sns.countplot(data=df, x='Fraud_Type', order=order)
        plt.title("Distribution of Fraud Types")
        plt.xlabel("Fraud Type")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()

    # Example: Correlation Heatmap for numeric columns
    num_cols = df.select_dtypes(include=[float, int]).columns.tolist()
    plt.figure(figsize=(12,10))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()
