# scripts/train.py

import numpy as np
import pandas as pd
import pickle
import xgboost as xgb

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import preprocess_and_split

def train_models(csv_file_path,
                 output_iso="models/iso_model.pkl",
                 output_xgb="models/xgb_model.pkl",
                 output_le="models/label_encoder.pkl"):
    """
    1) Preprocess & split data (train/test).
    2) Train Inverted IsolationForest on FRAUD subset.
    3) Train XGBoost on balanced data (SMOTE).
    4) Evaluate XGBoost on test set.
    5) Optionally evaluate Isolation Forest on entire dataset.
    6) Save models.
    """

    # -----------------------------
    # A. Preprocess & Split
    # -----------------------------
    X_train, X_test, y_train, y_test, X_train_bal, y_train_bal, le = preprocess_and_split(csv_file_path)
    print(f"\nData shapes:\n  X_train={X_train.shape}, y_train={y_train.shape}\n"
          f"  X_test={X_test.shape}, y_test={y_test.shape}\n"
          f"  X_train_bal={X_train_bal.shape}, y_train_bal={y_train_bal.shape}")

    # -----------------------------
    # B. Inverted IsolationForest on FRAUD
    # -----------------------------
    # Identify the label index for "No Fraud"
    no_fraud_label = (le.classes_ == "No Fraud").nonzero()[0]
    if len(no_fraud_label) > 0:
        no_fraud_label = no_fraud_label[0]
    else:
        no_fraud_label = None

    # Create FRAUD-only subset (since "No Fraud" is outlier)
    if no_fraud_label is not None:
        fraud_mask = (y_train_bal != no_fraud_label)
    else:
        fraud_mask = np.ones_like(y_train_bal, dtype=bool)  # everything is fraud if no 'No Fraud'

    X_fraud = X_train_bal[fraud_mask]

    iso = IsolationForest(
        n_estimators=100,
        contamination=0.05,  # ~5% no-fraud outliers
        random_state=42
    )
    iso.fit(X_fraud)
    print("\nIsolation Forest (Inverted) trained on FRAUD subset.")

    # -----------------------------
    # C. Train XGBoost (softprob)
    # -----------------------------
    num_classes = len(le.classes_)
    clf_prob = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=num_classes,
        eval_metric='mlogloss',
        random_state=42
    )
    clf_prob.fit(X_train_bal, y_train_bal)
    print("\nXGBoost (softprob) trained on balanced data.")

    # -----------------------------
    # D. Evaluate XGBoost on Test Set
    # -----------------------------
    y_pred = clf_prob.predict(X_test)
    xgb_acc = accuracy_score(y_test, y_pred)
    print(f"\nXGBoost Test Accuracy: {xgb_acc:.4f}")

    print("\nXGBoost Classification Report:")
    print(classification_report(y_test, y_pred))

    cm_xgb = confusion_matrix(y_test, y_pred)
    print("\nXGBoost Confusion Matrix:")
    print(cm_xgb)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.inverse_transform(np.arange(num_classes)),
                yticklabels=le.inverse_transform(np.arange(num_classes)))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("XGBoost Confusion Matrix (Test Set)")
    plt.show()

    # -----------------------------
    # E. Evaluate Isolation Forest (Inverted) on Entire Dataset (Optional)
    # -----------------------------
    # If you want to see how the inverted IF does by default:
    #   1 => inlier => "Fraud"
    #   -1 => outlier => "No Fraud"
    print("\nEvaluating Isolation Forest (Inverted) on entire dataset...")

    # We must ensure the entire dataset is aligned with the same columns
    # used for training X_fraud
    # (We can reuse X_train, X_test, or build a new X_all from df).
    # But let's do a quick approach:
    X_all = pd.concat([X_train, X_test], ignore_index=True)
    y_all = np.concatenate([y_train, y_test], axis=0)

    iso_preds = iso.predict(X_all)  # 1 => fraud, -1 => no fraud
    iso_preds_binary = np.where(iso_preds == 1, 1, 0)

    # ground truth: 0 => "No Fraud", 1 => "Fraud"
    # We can map from label-encoded y_all to 0/1
    # If no_fraud_label is None, we skip. But presumably it's not None.
    y_all_binary = np.where(y_all == no_fraud_label, 0, 1)

    iso_acc = accuracy_score(y_all_binary, iso_preds_binary)
    print(f"Inverted IF Accuracy (entire dataset): {iso_acc:.4f}")

    print("\nInverted IF Classification Report (entire dataset):")
    print(classification_report(y_all_binary, iso_preds_binary, target_names=["No Fraud","Fraud"]))

    cm_iso = confusion_matrix(y_all_binary, iso_preds_binary)
    print("\nInverted IF Confusion Matrix (entire dataset):")
    print(cm_iso)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm_iso, annot=True, fmt='d', cmap='Blues',
                xticklabels=["No Fraud (Pred)", "Fraud (Pred)"],
                yticklabels=["No Fraud (True)", "Fraud (True)"])
    plt.title("Inverted Isolation Forest Confusion Matrix (Entire Dataset)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    # -----------------------------
    # F. Save Models
    # -----------------------------
    with open(output_iso, "wb") as f:
        pickle.dump(iso, f)
    with open(output_xgb, "wb") as f:
        pickle.dump(clf_prob, f)
    with open(output_le, "wb") as f:
        pickle.dump(le, f)

    print(f"\nModels saved to: {output_iso}, {output_xgb}, {output_le}")
    print("Training complete!")

if __name__ == "__main__":
    csv_file = "data/Augmented_Fraud_Dataset_Final_Updated.csv"
    train_models(csv_file)
