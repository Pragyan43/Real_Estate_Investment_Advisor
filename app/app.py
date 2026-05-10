import streamlit as st
import pandas as pd
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide"
)

# =====================================
# CUSTOM CSS
# =====================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f5f7fa;
    }

    .title {
        font-size: 72px;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 10px;
    }

    .subtitle {
        font-size: 34px;
        color: #374151;
        text-align: center;
        margin-bottom: 40px;
    }

    .section-title {
        font-size: 32px;
        font-weight: bold;
        color: #111827;
        margin-top: 30px;
    }

    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =====================================
# LOAD MODELS
# =====================================

classifier = pickle.load(
    open("models/classification_model.pkl", "rb")
)

regressor = pickle.load(
    open("models/regression_model.pkl", "rb")
)

# =====================================
# LOAD DATASET
# =====================================

df = pd.read_csv("data/cleaned_real_estate.csv")

# =====================================
# HEADER
# =====================================

st.markdown(
    '<p class="title">🏠 Real Estate Investment Advisor</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Predict Property Profitability & Future Value using Machine Learning</p>',
    unsafe_allow_html=True
)

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("📌 About Project")

st.sidebar.info(
    """
    This AI-powered system helps users:

    ✅ Predict profitable property investments

    ✅ Estimate future property prices

    ✅ Analyze real estate market trends

    ✅ Generate ROI insights
    """
)

st.sidebar.success("Technologies Used")

st.sidebar.write(
    """
    - Python
    - Streamlit
    - Machine Learning
    - Random Forest
    - MLflow
    - Pandas
    - Matplotlib
    - Seaborn
    """
)

# =====================================
# MARKET INSIGHTS DASHBOARD
# =====================================

st.markdown(
    '<p class="section-title">📊 Real Estate Market Insights</p>',
    unsafe_allow_html=True
)

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        "Average Property Price",
        f"₹ {df['Price_in_Lakhs'].mean():.2f} Lakhs"
    )

with metric2:
    st.metric(
        "Average Property Size",
        f"{df['Size_in_SqFt'].mean():.0f} SqFt"
    )

with metric3:
    st.metric(
        "Average Price per SqFt",
        f"₹ {df['Price_per_SqFt'].mean():.2f}"
    )

# =====================================
# CHARTS SECTION
# =====================================

chart_col1, chart_col2 = st.columns(2)

# -------------------------------------
# Price Distribution
# -------------------------------------

with chart_col1:

    st.subheader("📈 Property Price Distribution")

    fig1, ax1 = plt.subplots()

    ax1.hist(
        df["Price_in_Lakhs"],
        bins=20
    )

    ax1.set_xlabel("Price in Lakhs")
    ax1.set_ylabel("Number of Properties")

    st.pyplot(fig1)

# -------------------------------------
# BHK Distribution
# -------------------------------------

with chart_col2:

    st.subheader("🏢 BHK Distribution")

    fig2, ax2 = plt.subplots()

    df["BHK"].value_counts().sort_index().plot(
        kind="bar",
        ax=ax2
    )

    ax2.set_xlabel("BHK")
    ax2.set_ylabel("Count")

    st.pyplot(fig2)

# =====================================
# HEATMAP
# =====================================

st.subheader("🔥 Feature Correlation Heatmap")

fig3, ax3 = plt.subplots(figsize=(12, 6))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="Blues",
    ax=ax3
)

st.pyplot(fig3)

# =====================================
# INPUT SECTION
# =====================================

st.markdown(
    '<p class="section-title">📋 Enter Property Details</p>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    bhk = st.slider(
        "Number of BHK",
        1,
        10,
        2
    )

    size = st.number_input(
        "Size in SqFt",
        min_value=100,
        value=1200
    )

    price = st.number_input(
        "Current Price (Lakhs)",
        min_value=1.0,
        value=50.0
    )

    age = st.slider(
        "Age of Property",
        0,
        50,
        5
    )

with col2:

    schools = st.slider(
        "Nearby Schools",
        0,
        10,
        2
    )

    hospitals = st.slider(
        "Nearby Hospitals",
        0,
        10,
        2
    )

    parking = st.slider(
        "Parking Spaces",
        0,
        5,
        1
    )

    furnished = st.selectbox(
        "Furnished Status",
        ["Fully Furnished", "Semi Furnished", "Unfurnished"]
    )

# =====================================
# FEATURE ENGINEERING
# =====================================

price_per_sqft = price / size

# =====================================
# PROPERTY SUMMARY
# =====================================

st.markdown(
    '<p class="section-title">📊 Property Summary</p>',
    unsafe_allow_html=True
)

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:
    st.metric(
        "Price per SqFt",
        f"₹ {price_per_sqft:.2f}"
    )

with summary_col2:
    st.metric(
        "Property Age",
        f"{age} Years"
    )

with summary_col3:
    st.metric(
        "BHK",
        bhk
    )

# =====================================
# INPUT DATAFRAME
# =====================================

input_data = pd.DataFrame({
    "BHK": [bhk],
    "Size_in_SqFt": [size],
    "Price_in_Lakhs": [price],
    "Price_per_SqFt": [price_per_sqft],
    "Age_of_Property": [age],
    "Nearby_Schools": [schools],
    "Nearby_Hospitals": [hospitals],
    "Parking_Space": [parking]
})

# =====================================
# PREDICTION BUTTON
# =====================================

if st.button("🚀 Predict Investment"):

    with st.spinner("Analyzing property data using Machine Learning..."):
        time.sleep(2)

    # ---------------------------------
    # Predictions
    # ---------------------------------

    investment_prediction = classifier.predict(input_data)[0]

    future_price_prediction = regressor.predict(input_data)[0]

    # ---------------------------------
    # RESULTS
    # ---------------------------------

    st.markdown(
        '<p class="section-title">📈 Prediction Results</p>',
        unsafe_allow_html=True
    )

    result_col1, result_col2 = st.columns(2)

    # ---------------------------------
    # Investment Result
    # ---------------------------------

    with result_col1:

        st.subheader("💰 Investment Analysis")

        if investment_prediction == 1:

            st.success(
                "✅ This Property is a GOOD Investment"
            )

            st.balloons()

        else:

            st.error(
                "❌ This Property is NOT a Good Investment"
            )

    # ---------------------------------
    # Future Price
    # ---------------------------------

    with result_col2:

        st.subheader("🏡 Future Price Prediction")

        st.info(
            f"Estimated Property Price after 5 Years: ₹ {future_price_prediction:.2f} Lakhs"
        )

    # ---------------------------------
    # ROI ANALYSIS
    # ---------------------------------

    profit = future_price_prediction - price

    roi = (profit / price) * 100

    st.markdown(
        '<p class="section-title">📊 ROI Analysis</p>',
        unsafe_allow_html=True
    )

    roi_col1, roi_col2 = st.columns(2)

    with roi_col1:

        st.metric(
            "Estimated Profit",
            f"₹ {profit:.2f} Lakhs"
        )

    with roi_col2:

        st.metric(
            "Estimated ROI",
            f"{roi:.2f}%"
        )

    # ---------------------------------
    # AI RECOMMENDATION
    # ---------------------------------

    st.markdown(
        '<p class="section-title">🤖 AI Recommendation</p>',
        unsafe_allow_html=True
    )

    if roi > 40:

        st.success(
            "Strong investment opportunity with high expected return."
        )

    elif roi > 20:

        st.warning(
            "Moderate investment opportunity with decent returns."
        )

    else:

        st.error(
            "Low return prediction. Consider other properties."
        )

# =====================================
# FOOTER
# =====================================

st.write("---")

st.markdown(
    """
    <center>
    <h3>Developed using Machine Learning, Streamlit & MLflow</h3>
    <p>Real Estate Investment Advisor Project</p>
    </center>
    """,
    unsafe_allow_html=True
)