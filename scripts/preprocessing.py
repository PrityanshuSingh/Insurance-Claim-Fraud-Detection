import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

def load_data(csv_file_path):
    """Load raw data from CSV into a DataFrame."""
    df = pd.read_csv(csv_file_path)
    return df

def clean_data(df):
    """
    Perform cleaning steps:
      1) Drop unnecessary columns
      2) Convert date columns
      3) Convert numeric columns
      4) Create derived features (Policy Tenure, etc.)
      5) Replace zero in sum assured
      6) Compute ratio columns
      7) Replace inf, drop rows with NaN in ratio columns
    """
    # 1. Drop unnecessary columns
    df_cleaned = df.drop(columns=["Dummy Policy No", "Bank code"], errors="ignore")

    # 2. Convert date columns to datetime
    date_columns = ["POLICYRISKCOMMENCEMENTDATE", "Date of Death", "INTIMATIONDATE"]
    for col in date_columns:
        if col in df_cleaned.columns:
            df_cleaned[col] = pd.to_datetime(df_cleaned[col],format="%Y-%m-%d",errors="coerce")

    # 3. Convert numeric columns
    numeric_cols = ["POLICY SUMASSURED", "Premium", "Annual Income"]
    for col in numeric_cols:
        df_cleaned[col] = (
            df_cleaned[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce")

    # 4. Create derived features
    if "Date of Death" in df_cleaned.columns and "POLICYRISKCOMMENCEMENTDATE" in df_cleaned.columns:
        df_cleaned["Policy Tenure (Days)"] = (
            df_cleaned["Date of Death"] - df_cleaned["POLICYRISKCOMMENCEMENTDATE"]
        ).dt.days
    if "INTIMATIONDATE" in df_cleaned.columns and "Date of Death" in df_cleaned.columns:
        df_cleaned["Claim Intimation Lag (Days)"] = (
            df_cleaned["INTIMATIONDATE"] - df_cleaned["Date of Death"]
        ).dt.days

    # 5. Replace zero in POLICY SUMASSURED with NaN (if invalid)
    if "POLICY SUMASSURED" in df_cleaned.columns:
        df_cleaned["POLICY SUMASSURED"] = df_cleaned["POLICY SUMASSURED"].replace(0, np.nan)

    # 6. Compute ratio columns
    if "Premium" in df_cleaned.columns and "POLICY SUMASSURED" in df_cleaned.columns:
        df_cleaned["Premium-to-Sum Assured Ratio"] = (
            df_cleaned["Premium"] / df_cleaned["POLICY SUMASSURED"]
        )
        df_cleaned["Income-to-Sum Assured Ratio"] = (
            df_cleaned["Annual Income"] / df_cleaned["POLICY SUMASSURED"]
        )

    # 7. Replace infinities, drop rows with NaN in ratio columns
    df_cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_cleaned.dropna(
        subset=["Premium-to-Sum Assured Ratio", "Income-to-Sum Assured Ratio"],
        inplace=True
    )

    return df_cleaned

def encode_data(df_cleaned):
    """
    Perform dummy encoding for categorical columns,
    handle 'Fraud_Type' column, drop date columns, etc.
    """
    # 8. Encode categorical columns
    categorical_columns = [
        "NOMINEE_RELATION",
        "OCCUPATION",
        "PREMIUMPAYMENTMODE",
        "HOLDERMARITALSTATUS",
        "CHANNEL",
        "Product Type"
    ]
    df_encoded = pd.get_dummies(df_cleaned, columns=categorical_columns, drop_first=True)

    # 9. Handle target variable
    df_encoded["Fraud_Type"] = df_encoded["Fraud Category"].fillna("No Fraud")
    df_encoded["Fraudulent"] = 1
    df_encoded.drop(columns=["Fraud Category", "STATUS", "SUB_STATUS"], inplace=True, errors="ignore")

    # 10. Drop original date columns
    date_columns = ["POLICYRISKCOMMENCEMENTDATE", "Date of Death", "INTIMATIONDATE"]
    df_encoded.drop(columns=date_columns, inplace=True, errors="ignore")

    return df_encoded

def preprocess_and_split(csv_file_path, test_size=0.3, rare_threshold=5, random_state=42):
    """
    1) Load raw CSV data
    2) Clean data (clean_data)
    3) Encode data (encode_data)
    4) Merge rare classes in Fraud_Type
    5) Label Encode
    6) Train-Test Split
    7) Fill NaN with median
    8) SMOTE for multi-class
    9) Return X_train, X_test, y_train, y_test, X_train_bal, y_train_bal, label_encoder
    """
    # 1) Load data
    df = load_data(csv_file_path)
    # 2) Clean
    df_cleaned = clean_data(df)
    # 3) Encode
    df_encoded = encode_data(df_cleaned)

    # 4) Prepare features
    features = df_encoded.select_dtypes(include=[np.number]).drop(['Fraudulent'], axis=1, errors="ignore")
    # Drop columns with excessive missing (like Policy Tenure, Claim Intimation Lag)
    features.drop(['Policy Tenure (Days)', 'Claim Intimation Lag (Days)'], axis=1, errors='ignore', inplace=True)

    # 5) Merge rare classes in Fraud_Type
    target_raw = df_encoded['Fraud_Type']
    freq = target_raw.value_counts()
    target_raw = target_raw.apply(lambda x: "Other" if freq[x] < rare_threshold else x)

    # 6) Label Encode
    le = LabelEncoder()
    target = le.fit_transform(target_raw)

    # 7) Train-Test Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        features, target,
        test_size=test_size,
        random_state=random_state,
        stratify=target
    )

    # 8) Fill NaN with median
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_test.median())

    # 9) SMOTE for multi-class
    from imblearn.over_sampling import SMOTE
    min_samples = np.min(pd.Series(y_train).value_counts())
    k_neighbors = max(1, min(5, min_samples - 1))
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test, X_train_bal, y_train_bal, le
