# scripts/hybrid_predict.py

import pandas as pd
import numpy as np

def hybrid_predict_proba_inverted(
    sample_features,
    iso_model,
    xgb_model,
    label_encoder,
    common_cols,
    X_train,
    decision_threshold=0.02,  # <--- Named 'decision_threshold' here
    num_classes=None
):
    """
    Inverted Isolation Forest + XGBoost Hybrid:
      1) Align columns => fill missing using X_train medians
      2) iso_model.decision_function(...) => if score >= decision_threshold => FRAUD => pass to XGBoost
         else => NO FRAUD
      3) Return (pred_label, probs_df, is_fraud_boolean)

    Parameters:
      sample_features (pd.DataFrame): A single-row or multiple-row DataFrame of raw inputs.
      iso_model: The trained IsolationForest (inverted) model.
      xgb_model: The trained XGBoost softprob model.
      label_encoder: The fitted LabelEncoder for fraud types.
      common_cols: The exact columns your model expects (list or Index).
      X_train: A reference DataFrame for median filling. (Used for fillna.)
      decision_threshold (float): If iso_score >= threshold => FRAUD.
      num_classes (int): Number of fraud-type classes. If None, inferred from label_encoder.

    Returns:
      (pred_label, probs_df, is_fraud)
        pred_label: A single predicted label if FRAUD, or "No Fraud" if not.
        probs_df: Probability distribution as a DataFrame.
        is_fraud: Boolean, True if FRAUD, False if NO FRAUD.
    """

    # 1) Align columns with training columns
    sample_features = sample_features[common_cols].copy()

    # 2) Fill missing using median from X_train
    sample_features = sample_features.fillna(X_train[common_cols].median())

    # 3) Isolation Forest decision score
    iso_score = iso_model.decision_function(sample_features)[0]
    print(f"Inverted Isolation Forest decision score: {iso_score:.4f}")

    # 4) If iso_score >= threshold => FRAUD => use XGBoost
    if iso_score >= decision_threshold:
        # XGBoost classification
        probs = xgb_model.predict_proba(sample_features)

        # If num_classes not specified, infer from label_encoder
        if num_classes is None:
            num_classes = len(label_encoder.classes_)

        # Build probability DataFrame
        class_names = label_encoder.inverse_transform(np.arange(num_classes))
        probs_df = pd.DataFrame(probs, columns=class_names)

        # Predicted label
        pred_label_num = xgb_model.predict(sample_features)
        pred_label = label_encoder.inverse_transform(pred_label_num)[0]

        return pred_label, probs_df, True  # FRAUD
    else:
        # NO FRAUD => probability dict with "No Fraud"=1.0
        if num_classes is None:
            num_classes = len(label_encoder.classes_)

        class_names = label_encoder.inverse_transform(np.arange(num_classes))
        prob_dict = {cls: 0.0 for cls in class_names}
        prob_dict["No Fraud"] = 1.0

        return "No Fraud", pd.DataFrame([prob_dict]), False
