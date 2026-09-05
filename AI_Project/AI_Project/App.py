import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE SETUP ----------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("outputs/models/xgboost.pkl")

# ---------------- HEADER ----------------
st.title("📊 Customer Churn Prediction System")
st.write(
    "A machine learning system that predicts customer churn risk "
    "and provides personalized retention recommendations."
)

st.divider()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("🤖 About the Model")

    st.write("**Algorithm:** XGBoost")
    st.write("**Accuracy:** 80.7%")
    st.write("**ROC-AUC:** 84.4%")
    st.write("**Dataset:** Telco Customer Churn")
    st.write("**Customers:** 7,043")

    st.divider()

    st.subheader("Risk Levels")
    st.write("🟢 **Low Risk:** < 30%")
    st.write("🟠 **Medium Risk:** 30–59%")
    st.write("🔴 **High Risk:** ≥ 60%")

# ---------------- INPUT SECTION ----------------
st.header("👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])

    senior = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    partner = st.selectbox("Partner", ["Yes", "No"])

    dependents = st.selectbox("Dependents", ["Yes", "No"])

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )

    phone_service = st.selectbox("Phone Service", ["Yes", "No"])

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

with col2:
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

with col3:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=10000.0,
        value=1000.0
    )

st.divider()

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Customer Churn", use_container_width=True):

    customer = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    probability = model.predict_proba(customer)[0][1]
    risk_score = probability * 100

    # Risk classification
    if risk_score < 30:
        risk_level = "Low Risk"
        recommendation = (
            "Customer appears relatively stable. "
            "Continue normal loyalty and customer-service activities."
        )

    elif risk_score < 60:
        risk_level = "Medium Risk"
        recommendation = (
            "Consider a personalized plan, loyalty benefit, "
            "or service improvement offer."
        )

    else:
        risk_level = "High Risk"

        if contract == "Month-to-month":
            recommendation = (
                "Offer a discounted long-term contract to encourage retention."
            )

        elif tech_support == "No" and internet_service == "Fiber optic":
            recommendation = (
                "Offer technical support and service assistance."
            )

        elif monthly_charges > 70:
            recommendation = (
                "Offer a personalized plan or billing discount."
            )

        else:
            recommendation = (
                "Provide loyalty rewards and targeted retention benefits."
            )

    # ---------------- RESULTS ----------------
    st.divider()
    st.header("🎯 Prediction Results")

    result1, result2, result3 = st.columns(3)

    with result1:
        st.metric(
            "Churn Probability",
            f"{risk_score:.1f}%"
        )

    with result2:
        st.metric(
            "Risk Level",
            risk_level
        )

    with result3:
        priority_score = probability * monthly_charges

        st.metric(
            "Retention Priority",
            f"{priority_score:.2f}"
        )

    # Risk message
    if risk_level == "High Risk":
        st.error(
            "🔴 HIGH RISK — This customer may be likely to churn."
        )

    elif risk_level == "Medium Risk":
        st.warning(
            "🟠 MEDIUM RISK — This customer may require additional attention."
        )

    else:
        st.success(
            "🟢 LOW RISK — This customer appears relatively stable."
        )

    # Recommendation
    st.subheader("💡 Recommended Retention Action")
    st.info(recommendation)

    # Customer summary
    st.subheader("📋 Customer Summary")

    summary = pd.DataFrame({
        "Attribute": [
            "Contract",
            "Internet Service",
            "Tenure",
            "Monthly Charges",
            "Tech Support"
        ],
        "Value": [
            contract,
            internet_service,
            f"{tenure} months",
            f"${monthly_charges:.2f}",
            tech_support
        ]
    })

    st.table(summary)

st.divider()

st.caption(
    "Customer Churn Prediction Project | XGBoost Machine Learning Model"
)