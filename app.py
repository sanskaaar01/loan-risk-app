import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime
import random

# 🎯 Page config
st.set_page_config(page_title="📊 Loan Risk Predictor", layout="centered")

# 📦 Load model and scaler
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "..", "database", "model", "loan_default_model.pkl")
scaler_path = os.path.join(base_dir, "..", "database", "model", "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# 🔐 User credentials
users = {
    "sanskar": "pass123",
    "admin": "adminpass"
}

# 🧠 Session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# 🔐 Login screen
if not st.session_state.logged_in:
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome, {username}!")
        else:
            st.error("Invalid credentials")
    st.stop()

# 💬 Feedback responses
low_risk_responses = [
    "✅ Loan can be approved with confidence.",
    "🟢 Applicant shows strong financial stability.",
    "👍 Low risk detected. Go ahead.",
    "✅ All signs point to a safe approval.",
    "🟢 This applicant is a green flag."
]

mid_risk_responses = [
    "🟠 Moderate risk. Review documents carefully.",
    "⚖️ Loan approval possible, but needs caution.",
    "🟠 Applicant may need additional verification.",
    "🤔 Not risky, but not entirely safe either.",
    "⚠️ Some indicators suggest caution."
]

high_risk_responses = [
    "🔴 High risk detected. Loan not recommended.",
    "⚠️ Applicant is likely to default.",
    "🚫 Approval is risky. Reconsider.",
    "🔴 Financial profile is concerning.",
    "🚫 Consider rejecting or requesting collateral."
]

# 🏦 App title
st.title("📊 Loan Default Risk Predictor")
st.markdown(f"Welcome, **{st.session_state.username}**! Enter applicant details below to assess loan risk.")

# 📝 Input form
age = st.number_input("🎂 Applicant Age", min_value=18, max_value=100, value=30)
education = st.selectbox("🎓 Education Level", ["Graduate", "Not Graduate"])
proof_submitted = st.selectbox("📄 Proof of Income Submitted", ["Yes", "No"])
loan_amount = st.number_input("💰 Loan Amount (₹)", min_value=1000, value=150000)
asset_cost = st.number_input("🏠 Asset Cost (₹)", min_value=1000, value=250000)
no_of_loans = st.number_input("📑 Number of Existing Loans", min_value=0, value=1)
last_delinq_none = st.selectbox("🧾 No Delinquency History", ["Yes", "No"])

# 🔍 Prediction logic
if st.button("🚀 Predict Loan Risk"):
    edu_map = {'Graduate': 1, 'Not Graduate': 0}
    proof_map = {'Yes': 1, 'No': 0}
    delinq_map = {'Yes': 1, 'No': 0}

    input_data = np.array([[
        age,
        edu_map[education],
        proof_map[proof_submitted],
        loan_amount,
        asset_cost,
        no_of_loans,
        delinq_map[last_delinq_none]
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # 🧠 Risk feedback
    if probability >= 0.75:
        feedback = random.choice(high_risk_responses)
        st.markdown("### 🔴 High Risk")
        st.error(feedback)
        tier = "Tier 3 – High Risk"
    elif 0.4 <= probability < 0.75:
        feedback = random.choice(mid_risk_responses)
        st.markdown("### 🟠 Medium Risk")
        st.warning(feedback)
        tier = "Tier 2 – Moderate Risk"
    else:
        feedback = random.choice(low_risk_responses)
        st.markdown("### 🟢 Low Risk")
        st.success(feedback)
        tier = "Tier 1 – Low Risk"

    st.markdown(f"**📈 Model Confidence:** `{probability:.2f}`")
    st.markdown(f"**📊 Risk Tier:** {tier}")

    # 📝 Log activity
    log_entry = {
        "user": st.session_state.username,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "age": age,
        "education": education,
        "proof_submitted": proof_submitted,
        "loan_amount": loan_amount,
        "asset_cost": asset_cost,
        "no_of_loans": no_of_loans,
        "last_delinq_none": last_delinq_none,
        "risk_tier": tier,
        "confidence": round(probability, 2)
    }

    log_df = pd.DataFrame([log_entry])
    log_df.to_csv("user_logs.csv", mode="a", header=not os.path.exists("user_logs.csv"), index=False)

# 📁 Admin view
if st.session_state.username == "admin":
    st.markdown("---")
    st.markdown("### 📁 User Activity Logs")
    if os.path.exists("user_logs.csv"):
        logs = pd.read_csv("user_logs.csv")
        st.dataframe(logs)
    else:
        st.info("No logs available yet.")

# 🧾 Footer
st.markdown("---")
st.caption("🛠️ Built by Sanskar Bhosle • 🤖 Powered by Machine Learning • 🔐 Multi-user Enabled • 📦 v1.1")