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
        <h1 style='font-size: 60px; color: #002147;'>FairMark: AI-Powered Trustworthy Marketing Automation Platform</h1>
    </div>
""", unsafe_allow_html=True)

# === Hero Image & Introduction ===
col1, col2 = st.columns([3, 2])
with col1:
    image_path = "C:/Users/anupr/OneDrive/Documents/epsilon/OIP.webp"
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    else:
        st.warning("Banner image not found.")

with col2:
    st.markdown("""
        <h1 style='font-size: 50px;'>Smart decisioning<br>and real insights,<br>wherever you are</h1>
        <p style='font-size:25px;'>
            FairMark empowers ethical marketing teams with transparent, trustworthy AI tools.
            Our real-time automation platform lets you detect bias, explain predictions,
            enforce compliance, and mitigate risks — all from a unified dashboard.
        </p>
    """, unsafe_allow_html=True)

st.markdown("---")

# === Why Responsible AI ===
st.markdown("<h2 style='font-size:50px'>AI That Builds Trust</h2>", unsafe_allow_html=True)

# === Section 1 ===
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
    <h3 style='font-size:45px'>📦 <strong>Smarter shopper interactions</strong></h3>
    <p style='font-size:25px;'>
    FairMark equips marketing teams with AI-powered decision systems that understand shopper behavior in real time. Instead of generic campaigns, it prioritizes high-intent individuals, increasing the relevance of offers and reducing wasted outreach.<br>
    - Uses predictive analytics to identify the right moment for engagement<br>
    - Enhances customer experience with personalized messaging<br>
    - Reduces ad fatigue by avoiding irrelevant content
    </p>
    """, unsafe_allow_html=True)
with col2:
    img1 = "C:/Users/anupr/OneDrive/Documents/epsilon/shopper_insights.jpg"
    if os.path.exists(img1):
        st.image(img1, caption="Smarter Interactions")

# === Section 2 ===
col3, col4 = st.columns([2, 1])
with col3:
    st.markdown("""
    <h3 style='font-size:45px'>🎯 <strong>Honest shopper targeting and measurement</strong></h3>
    <p style='font-size:25px;'>
    FairMark’s proprietary CORE ID technology moves beyond unreliable tracking methods. Instead of anonymous or third-party data, it focuses on verified shoppers, ensuring precision and accountability.<br>
    - Guarantees you’re reaching actual people, not bots or duplicate profiles<br>
    - Enables robust measurement of campaign impact<br>
    - Improves ROI by sharpening targeting accuracy
    </p>
    """, unsafe_allow_html=True)
with col4:
    img2 = "C:/Users/anupr/OneDrive/Documents/epsilon/core_id.jpg"
    if os.path.exists(img2):
        st.image(img2, caption="Verified Shopper Targeting")

# === Section 3 ===
col5, col6 = st.columns([2, 1])
with col5:
    st.markdown("""
    <h3 style='font-size:45px'>🧩 <strong>A complete marketing trust stack</strong></h3>
    <p style='font-size:25px;'>
    Trust is no longer optional — it’s a competitive advantage. FairMark offers a full suite of ethical utilities that make compliance, transparency, and fairness built-in, not an afterthought.<br>
    - Seamless onboarding flows with built-in consent management<br>
    - AI explainability tools so marketers understand why decisions are made<br>
    - Bias detection safeguards to prevent unfair targeting or exclusion
    </p>
    """, unsafe_allow_html=True)
with col6:
    img3 = "C:/Users/anupr/OneDrive/Documents/epsilon/trust_stack.jpg"
    if os.path.exists(img3):
        st.image(img3, caption="AI Trust Stack")
