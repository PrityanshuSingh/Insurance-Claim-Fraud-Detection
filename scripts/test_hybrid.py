# scripts/test_hybrid.py

import pickle
import numpy as np
import pandas as pd
from preprocessing import preprocess_and_split
from hybrid_predict import hybrid_predict_proba_inverted

def main():
    # 1) Preprocess & split again for test
    csv_file = "data/Augmented_Fraud_Dataset_Final_Updated.csv"
    X_train, X_test, y_train, y_test, X_train_bal, y_train_bal, le = preprocess_and_split(csv_file)

    # 2) Load models
    with open("models/iso_model.pkl", "rb") as f:
        iso = pickle.load(f)
    with open("models/xgb_model.pkl", "rb") as f:
        clf_prob = pickle.load(f)

    # 3) Random sample from X_test
    random_idx = np.random.choice(X_test.index)
    sample = X_test.loc[[random_idx]]
    print("\nRandom sample from X_test:")
    print(sample)

    print("\n columns:", X_train.columns)
    # 4) Call hybrid_predict_proba_inverted
    pred_label, probs_df, is_fraud = hybrid_predict_proba_inverted(
        sample_features=sample,
        iso_model=iso,
        xgb_model=clf_prob,
        label_encoder=le,
        common_cols=X_train.columns,
        X_train=X_train,
        decision_threshold=0.02,
        num_classes=len(le.classes_)
    )

    if is_fraud:
        print("Fraud Detected:", pred_label)
    else:
        print("No Fraud Detected")

    print("Probabilities:")
    print(probs_df)

if __name__ == "__main__":
    main()
