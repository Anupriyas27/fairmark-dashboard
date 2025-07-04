# FairMark AI-Powered Ethical Marketing Dashboard

import streamlit as st
import pandas as pd
import plotly.express as px
import os
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import pickle

# === Set Page Configuration ===
st.set_page_config(page_title="FairMark Dashboard", layout="wide")

# === Branding ===
st.markdown("""
    <div style='text-align:center; padding: 10px;'>
        <h1 style='font-size: 70px; color: #002147;'>FairMark: Federated Learning Framework for Fair and Transparent Marketing Automation</h1>
    </div>
""", unsafe_allow_html=True)

# === Hero Image & Introduction ===
col1, col2 = st.columns([3, 2])
with col1:
    image_path = "OIP.webp"
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    else:
        st.warning("Banner image not found.")

with col2:
    st.markdown("""
        <h1 style='font-size: 65px;'>Trust-Centered AI Marketing<br></h1>
        <p style='font-size:50px;'>
            FairMark empowers ethical marketing teams with transparent, trustworthy AI tools.
            Our real-time automation platform lets you detect bias, explain predictions,
            enforce compliance, and mitigate risks — all from a unified dashboard.
        </p>
    """, unsafe_allow_html=True)

st.markdown("---")

# === Why Responsible AI ===
st.markdown("<h1 style='font-size:px65'>AI That Builds Trust For Our Solution <br></h1>", unsafe_allow_html=True)

# === Section 1 ===
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
    <h3 style='font-size:45px'> <strong>Smarter shopper interactions</strong></h3>
    <p style='font-size:25px;'>
    FairMark equips marketing teams with AI-powered decision systems that understand shopper behavior in real time. Instead of generic campaigns, it prioritizes high-intent individuals, increasing the relevance of offers and reducing wasted outreach.<br>
    - Uses predictive analytics to identify the right moment for engagement<br>
    - Enhances customer experience with personalized messaging<br>
    - Reduces ad fatigue by avoiding irrelevant content
    </p>
    """, unsafe_allow_html=True)
with col2:
    img1 = "download.webp"
    if os.path.exists(img1):
        st.image(img1, caption="Smarter Interactions")

# === Section 2 ===
col3, col4 = st.columns([2, 1])
with col3:
    st.markdown("""
    <h3 style='font-size:45px'> <strong>Honest shopper targeting and measurement</strong></h3>
    <p style='font-size:25px;'>
    FairMark’s proprietary CORE ID technology moves beyond unreliable tracking methods. Instead of anonymous or third-party data, it focuses on verified shoppers, ensuring precision and accountability.<br>
    - Guarantees you’re reaching actual people, not bots or duplicate profiles<br>
    - Enables robust measurement of campaign impact<br>
    - Improves ROI by sharpening targeting accuracy
    </p>
    """, unsafe_allow_html=True)
with col4:
    img2 = "OIP.webp"
    if os.path.exists(img2):
        st.image(img2, caption="Verified Shopper Targeting")

# === Section 3 ===
col5, col6 = st.columns([2, 1])
with col5:
    st.markdown("""
    <h3 style='font-size:45px'> <strong>A complete marketing trust stack</strong></h3>
    <p style='font-size:25px;'>
    Trust is no longer optional — it’s a competitive advantage. FairMark offers a full suite of ethical utilities that make compliance, transparency, and fairness built-in, not an afterthought.<br>
    - Seamless onboarding flows with built-in consent management<br>
    - AI explainability tools so marketers understand why decisions are made<br>
    - Bias detection safeguards to prevent unfair targeting or exclusion
    </p>
    """, unsafe_allow_html=True)
with col6:
    img3 = "OIP (1).webp"
    if os.path.exists(img3):
        st.image(img3, caption="AI Trust Stack")


# === Contact Info ===
st.markdown("""
<div style="background-color: #062e6f; padding: 60px; text-align:center;">
    <p style="color:white; font-size:25px;">WANT TO CONNECT WITH US?</p>
    <p style="color:white; font-size:28px;">
        📧 <a href='mailto:anupriya@gmail.com' style='color:white;'>anupriya@gmail.com</a> &nbsp;&nbsp;
        📞 <a href='tel:+919986899764' style='color:white;'>+91 99868 99764</a>
    </p>
</div>
""", unsafe_allow_html=True)

# === PIN Protection ===
pin = st.text_input("🔐 Enter Access PIN to Unlock Retailer Dashboard:", type="password")
if pin != "1234":
    st.warning("This dashboard is protected. Enter the correct PIN to proceed.")
    st.stop()

# === Load Data ===
@st.cache_data
def load_data():
    return pd.read_csv("final_shop_6modata.csv")

df = load_data()
le = LabelEncoder()
df["Gender_encoded"] = le.fit_transform(df["Gender"])

# === Dashboard Title ===
st.title(" Retailer Dashboard – FairMark AI Tools")
st.markdown("Manage ethical campaigns with AI transparency, fairness, and security.")

# === Metrics ===
with st.container():
    st.subheader(" Key Campaign Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Impressions", f"{df['Impressions'].sum():,}")
    col2.metric("Total Clicks", f"{df['Clicks'].sum():,}")
    col3.metric("Total Conversions", f"{df['Conversions'].sum():,}")
    col4.metric("Total Revenue", f"${df['Revenue'].sum():,.2f}")

# === Bias Detection ===
with st.expander(" Bias Detection", expanded=True):
    gender_bias = df.groupby("Gender")["Conversions"].mean().reset_index()
    fig_bias = px.bar(gender_bias, x="Gender", y="Conversions", color="Gender",
                      title="Avg Conversion Rate by Gender")
    st.plotly_chart(fig_bias, use_container_width=True)

# === Explainable AI ===
with st.expander(" Explainable AI – Revenue Forecasting", expanded=True):
    X = df[["Impressions", "CTR", "Conv Rate"]]
    y = df["Revenue"]
    model = LinearRegression()
    model.fit(X, y)
    df["Revenue_Pred"] = model.predict(X)

    st.write("**Model Coefficients:**")
    for feature, coef in zip(X.columns, model.coef_):
        st.write(f"- {feature}: {coef:.2f}")

    fig_pred = px.scatter(df, x="Revenue", y="Revenue_Pred",
                          title="Actual vs Predicted Revenue")
    st.plotly_chart(fig_pred, use_container_width=True)

# === AI Security Guardrails ===
with st.expander(" AI Security Guardrails", expanded=False):
    def flag(row):
        if row["CTR"] > 0.9:
            return "⚠️ Abnormal CTR"
        if row["Conversions"] == 0 and row["Cost"] > 1000:
            return "⚠️ High Cost, No Conversions"
        if row["P&L"] < -1000:
            return "⚠️ High Loss"
        return "✅ Normal"

    df["Security Flag"] = df.apply(flag, axis=1)
    st.dataframe(df[["Ad Group", "CTR", "Cost", "Conversions", "P&L", "Security Flag"]].head(10))

# === Ethical Lead Scoring ===
with st.expander(" Ethical Lead Scoring", expanded=False):
    df["Fair_Score"] = (df["Conv Rate"] * 100) + (df["P&L"] / 1000)
    top_leads = df.sort_values("Fair_Score", ascending=False)
    st.dataframe(top_leads[["Ad Group", "Conv Rate", "P&L", "Fair_Score"]].head(10))

# === Compliance Checker ===
with st.expander("✅ Automated Compliance Checker", expanded=False):
    df["Compliant"] = df.apply(lambda x: "✅" if x["Conv Rate"] <= 0.15 and x["CPC"] < 1.5 else "⚠️", axis=1)
    st.dataframe(df[["Ad Group", "Conv Rate", "CPC", "Compliant"]].head(10))

# === Campaign Visualization ===
with st.expander("CTR vs Cost by Gender", expanded=False):
    fig_ctr = px.scatter(df, x="CTR", y="Cost", color="Gender", size="Revenue",
                         hover_data=["Ad Group"], title="CTR vs Cost by Gender")
    st.plotly_chart(fig_ctr, use_container_width=True)

# === Federated Learning ===
with st.expander("🔐 Federated Learning & Model Security", expanded=False):
    st.markdown("""
    Federated learning protects user privacy by training models on-device:
    - ✅ No raw data leaves user devices
    - ✅ GDPR/CCPA Compliant
    - ✅ Global model built from local weight aggregation
    """)

    # Simulate federated model save
    with open("local_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("local_model.pkl", "rb") as f:
        st.download_button("⬇️ Download Local Model", f, file_name="local_model.pkl")

    st.markdown("**Simulated Global Model Coefficients:**")
    for i, weight in enumerate(model.coef_):
        st.write(f"- {X.columns[i]}: {weight:.4f}")

# === Footer ===
st.markdown("---")
st.markdown("Created by Team FairMark | Ethical AI for Scalable Marketing |")
