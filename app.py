import streamlit as st
import pandas as pd
import numpy as np
import pickle

from scripts.hybrid_predict import hybrid_predict_proba_inverted

def load_models():
    with open("models/iso_model.pkl", "rb") as f:
        iso_model = pickle.load(f)
    with open("models/xgb_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open("models/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return iso_model, xgb_model, label_encoder

def main():
    st.title("Insurance Claim Fraud Detection App")

    # 1) Load Models
    iso_model, xgb_model, label_encoder = load_models()
    num_classes = len(label_encoder.classes_)

    st.subheader("Numeric Inputs related to insurance policy")
    st.write("Fill in the details below to check if the claim is fraudulent.")
    # We can arrange them in columns:
    col1, col2 = st.columns(2)

    with col1:
        assured_age = st.number_input("ASSURED_AGE", min_value=0.0, max_value=120.0, value=59.20221257)
        policy_sum_assured = st.number_input("POLICY SUMASSURED", min_value=0.0, value=1055799.635)
        premium = st.number_input("Premium", min_value=0.0, value=109360.6947)

    with col2:
        annual_income = st.number_input("Annual Income", min_value=0.0, value=43320.56362)
        policy_term = st.number_input("Policy Term", min_value=0.0, value=9.879896519)
        policy_payment_term = st.number_input("Policy Payment Term", min_value=0.0, value=2.824662067)

    st.write("---")
    st.subheader("Extra Fields in the insurance policy")

    # We can place checkboxes in columns as well
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Nominee**")
        nominee_son = st.checkbox("Son", value=True, key="nom_son")
        nominee_mother = st.checkbox("Mother", value=False, key="nom_mother")
        nominee_husband = st.checkbox("Husband", value=False, key="nom_husband")
        nominee_wife = st.checkbox("Wife", value=False, key="nom_wife")

        st.markdown("**Occupation**")
        occ_retired = st.checkbox("Retired", value=True, key="occ_retired")
        occ_service = st.checkbox("Service", value=False, key="occ_service")
        occ_business = st.checkbox("Business", value=False, key="occ_business")
        occ_profession = st.checkbox("Profession", value=False, key="occ_profession")

        st.markdown("**Product Type**")
        prod_ulip = st.checkbox("ULIP", value=True, key="prod_ulip")
        prod_traditional = st.checkbox("Traditional", value=False, key="prod_traditional")
        prod_variable = st.checkbox("Variable", value=False, key="prod_variable")
        # If you have "Product Type_Pension", add a checkbox here

    with col4:
        
        st.markdown("**Channel**")
        channel_retail = st.checkbox("Retail Agency", value=True, key="chan_retail")
        channel_banc = st.checkbox("Bancassurance", value=False, key="chan_banc")
        channel_instit = st.checkbox("Institutional Alliance", value=False, key="chan_instit")
        channel_mail = st.checkbox("Mail and Others", value=False, key="chan_mail")
        
        st.markdown("**Marital Status**")
        mar_married = st.checkbox("Married", value=True, key="mar_married")
        mar_single = st.checkbox("Single", value=False, key="mar_single")
        mar_divorced = st.checkbox("divorced", value=False, key="mar_divorced")
        mar_widowed = st.checkbox("widowed", value=False, key="mar_widowed")
        
        st.markdown("**Payment Mode**")
        pay_yearly = st.checkbox("Yearly", value=True, key="pay_yearly")
        pay_quarterly = st.checkbox("Quarterly", value=False, key="pay_quarterly")
        pay_monthly = st.checkbox("Monthly", value=False, key="pay_monthly")
        # pay_single = st.checkbox("Single", value=False, key="pay_single")
        

        st.markdown("**Requirement Flag**")
        req_medical = st.checkbox("Medical", value=True, key="req_medical")
        req_non_medical = st.checkbox("Non Medical", value=False, key="req_non_medical")
        

    # 2) Predict button
    if st.button("Predict Fraud?"):

        columns_needed = [
            "ASSURED_AGE",
            "POLICY SUMASSURED",
            "Premium",
            "Annual Income",
            "Policy Term",
            "Policy Payment Term",
            "Premium-to-Sum Assured Ratio",
            "Income-to-Sum Assured Ratio"
        ]

        # Create a single-row DataFrame
        data = {col: [0.0] for col in columns_needed}
        sample_df = pd.DataFrame(data)

        # Override numeric columns
        sample_df["ASSURED_AGE"] = assured_age
        sample_df["POLICY SUMASSURED"] = policy_sum_assured
        sample_df["Premium"] = premium
        sample_df["Annual Income"] = annual_income
        sample_df["Policy Term"] = policy_term
        sample_df["Policy Payment Term"] = policy_payment_term

        # Compute ratio columns
        if policy_sum_assured != 0:
            sample_df["Premium-to-Sum Assured Ratio"] = premium / policy_sum_assured
            sample_df["Income-to-Sum Assured Ratio"] = annual_income / policy_sum_assured
        else:
            sample_df["Premium-to-Sum Assured Ratio"] = 0.0
            sample_df["Income-to-Sum Assured Ratio"] = 0.0


        # Call the hybrid predictor
        pred_label, probs_df, is_fraud = hybrid_predict_proba_inverted(
            sample_features=sample_df,
            iso_model=iso_model,
            xgb_model=xgb_model,
            label_encoder=label_encoder,
            common_cols=sample_df.columns,
            X_train=sample_df,
            num_classes=num_classes
        )

        # Show results
        if is_fraud:
            print("Fraud Detected:", pred_label)
            st.write("**Fraud Detected:**", pred_label)
        else:
            print("No Fraud Detected")
            st.write("**No Fraud Detected**")

        st.write("Probabilities:")
        print("Probabilities:")
        print(probs_df)
        
        st.dataframe(probs_df)
        st.bar_chart(probs_df.T)
        
        

if __name__ == "__main__":
    main()
