import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide"
)

# =========================================
# CUSTOM CSS
# =========================================

st.markdown("""
<style>

.main {
    background-color: #0F172A;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1, h2, h3, h4, h5 {
    color: white;
}

p, span, div {
    color: #CBD5E1;
}

.metric-card {
    background: linear-gradient(135deg, #1E293B, #111827);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0px 6px 24px rgba(0,0,0,0.35);
    border: 1px solid rgba(255,255,255,0.05);
}

.metric-card h4 {
    color: #94A3B8;
    font-size: 16px;
}

.metric-card h2 {
    color: white;
    font-size: 32px;
}

.stButton>button {
    background: linear-gradient(135deg, #6366F1, #8B5CF6);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 0.7rem 1.5rem;
    font-weight: bold;
    width: 100%;
}

.stButton>button:hover {
    background: linear-gradient(135deg, #4F46E5, #7C3AED);
    color: white;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# HEADER
# =========================================

st.markdown("""
# 📊 Customer Churn Intelligence System

AI-powered customer retention analytics and churn prediction platform for strategic business decision-making.
""")

# =========================================
# LOAD DATA
# =========================================

df = pd.read_csv("data/churn.csv")

# =========================================
# CLEANING
# =========================================

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["TotalCharges"].fillna(
    df["TotalCharges"].median(),
    inplace=True
)

# =========================================
# FEATURE ENGINEERING
# =========================================

df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

df.drop("customerID", axis=1, inplace=True)

df_encoded = pd.get_dummies(
    df,
    drop_first=True
)

# =========================================
# SPLIT DATA
# =========================================

X = df_encoded.drop("Churn", axis=1)

y = df_encoded["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================
# TRAIN MODEL
# =========================================

model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# =========================================
# PREDICTION
# =========================================

probabilities = model.predict_proba(X_test)[:,1]

result_df = X_test.copy()

result_df["Actual_Churn"] = y_test.values

result_df["Churn_Probability"] = probabilities

# =========================================
# RISK SEGMENTATION
# =========================================

def risk_level(prob):

    if prob >= 0.75:
        return "🔴 High Risk"

    elif prob >= 0.30:
        return "🟠 Medium Risk"

    else:
        return "🟢 Low Risk"

result_df["Risk_Level"] = result_df[
    "Churn_Probability"
].apply(risk_level)

# =========================================
# KPI SECTION
# =========================================

total_customer = len(result_df)

high_risk = len(
    result_df[result_df["Risk_Level"] == "🔴 High Risk"]
)

avg_churn = f"{result_df['Churn_Probability'].mean() * 100:.2f}"

retention_priority = round(
    (high_risk / total_customer) * 100,
    2
)

st.markdown("## 📌 Executive Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Total Customers</h4>
        <h2>{total_customer}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h4>High Risk Customers</h4>
        <h2>{high_risk}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Average Churn Risk</h4>
        <h2>{avg_churn}%</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Retention Priority</h4>
        <h2>{retention_priority}%</h2>
    </div>
    """, unsafe_allow_html=True)

# =========================================
# RISK DISTRIBUTION
# =========================================

st.markdown("## 📊 Customer Risk Segmentation")

risk_counts = result_df["Risk_Level"].value_counts()

fig1 = px.pie(
    values=risk_counts.values,
    names=risk_counts.index,
    hole=0.65,
    color=risk_counts.index,
    color_discrete_map={
        "🔴 High Risk": "#EF4444",
        "🟠 Medium Risk": "#F59E0B",
        "🟢 Low Risk": "#22C55E"
    }
)

fig1.update_layout(
    paper_bgcolor="#0F172A",
    plot_bgcolor="#0F172A",
    font_color="white",
    title_font_size=24
)

st.plotly_chart(fig1, use_container_width=True)

# =========================================
# CHURN PROBABILITY DISTRIBUTION
# =========================================

st.markdown("## 📈 Churn Probability Analysis")

fig2 = px.histogram(
    result_df,
    x="Churn_Probability",
    nbins=30,
    color="Risk_Level",
    color_discrete_map={
        "🔴 High Risk": "#EF4444",
        "🟠 Medium Risk": "#F59E0B",
        "🟢 Low Risk": "#22C55E"
    }
)

fig2.update_layout(
    paper_bgcolor="#0F172A",
    plot_bgcolor="#0F172A",
    font_color="white"
)

st.plotly_chart(fig2, use_container_width=True)

# =========================================
# FEATURE IMPORTANCE
# =========================================

st.markdown("## 🧠 Top Feature Importance")

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
).head(10)

fig3 = px.bar(
    importance,
    x="Importance",
    y="Feature",
    orientation="h",
    color="Importance",
    color_continuous_scale="Blues"
)

fig3.update_layout(
    paper_bgcolor="#0F172A",
    plot_bgcolor="#0F172A",
    font_color="white"
)

st.plotly_chart(fig3, use_container_width=True)

# =========================================
# HIGH RISK CUSTOMER
# =========================================

st.markdown("## 🚨 High Risk Customer Monitoring")

high_risk_df = result_df[
    result_df["Risk_Level"] == "🔴 High Risk"
]

st.dataframe(
    high_risk_df[
        ["Churn_Probability", "Risk_Level"]
    ].head(20),
    use_container_width=True
)

# =========================================
# AI CUSTOMER PREDICTION SYSTEM
# =========================================

st.markdown("## 🤖 AI Customer Churn Prediction")

st.markdown("""
Predict churn probability for a new customer profile using the trained AI model.
""")

colA, colB = st.columns(2)

with colA:

    tenure = st.slider(
        "Customer Tenure (Months)",
        0,
        72,
        12
    )

    monthly_charges = st.slider(
        "Monthly Charges",
        0,
        150,
        70
    )

    total_charges = st.slider(
        "Total Charges",
        0,
        10000,
        2000
    )

    contract = st.selectbox(
        "Contract Type",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

with colB:

    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
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

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    senior = st.selectbox(
        "Senior Citizen",
        ["Yes", "No"]
    )

# =========================================
# PREDICT BUTTON
# =========================================

if st.button("🚀 Predict Churn Risk"):

    input_data = pd.DataFrame({
        "SeniorCitizen": [1 if senior == "Yes" else 0],
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    # create all missing columns

    for col in X_train.columns:

        if col not in input_data.columns:
            input_data[col] = 0

    # contract

    if contract == "One year":
        input_data["Contract_One year"] = 1

    elif contract == "Two year":
        input_data["Contract_Two year"] = 1

    # internet service

    if internet_service == "Fiber optic":
        input_data["InternetService_Fiber optic"] = 1

    elif internet_service == "No":
        input_data["InternetService_No"] = 1

    # payment method

    if payment_method == "Mailed check":
        input_data["PaymentMethod_Mailed check"] = 1

    elif payment_method == "Bank transfer (automatic)":
        input_data[
            "PaymentMethod_Bank transfer (automatic)"
        ] = 1

    elif payment_method == "Credit card (automatic)":
        input_data[
            "PaymentMethod_Credit card (automatic)"
        ] = 1

    # paperless billing

    if paperless_billing == "Yes":
        input_data["PaperlessBilling_Yes"] = 1

    # reorder columns

    input_data = input_data[
        X_train.columns
    ]

    # prediction

    prediction_prob = model.predict_proba(
        input_data
    )[0][1]

    # risk level

    if prediction_prob >= 0.75:

        risk = "🔴 High Risk"

        recommendation = """
        Immediate retention strategy recommended:
        - Offer discount or incentive
        - Customer engagement follow-up
        - Loyalty campaign
        """

    elif prediction_prob >= 0.30:

        risk = "🟠 Medium Risk"

        recommendation = """
        Moderate retention strategy:
        - Personalized recommendation
        - Engagement campaign
        """

    else:

        risk = "🟢 Low Risk"

        recommendation = """
        Customer retention status is stable.
        Maintain loyalty engagement.
        """

    # =========================================
    # RESULT
    # =========================================

    st.markdown("---")

    st.markdown("## 🎯 Prediction Result")

    colX, colY = st.columns(2)

    with colX:
        st.metric(
            "Churn Probability",
            f"{prediction_prob*100:.2f}%"
        )

    with colY:
        st.metric(
            "Risk Level",
            risk
        )

    # gauge chart

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction_prob * 100,
        title={'text': "Churn Risk Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#6366F1"},
            'steps': [
                {'range': [0, 30], 'color': "#22C55E"},
                {'range': [30, 75], 'color': "#F59E0B"},
                {'range': [75, 100], 'color': "#EF4444"}
            ]
        }
    ))

    fig_gauge.update_layout(
        paper_bgcolor='#0F172A',
        font_color='white'
    )

    st.plotly_chart(
        fig_gauge,
        use_container_width=True
    )

    # recommendation

    st.markdown("## 💡 Business Recommendation")

    st.success(recommendation)

# =========================================
# BUSINESS RECOMMENDATION
# =========================================

st.markdown("## 🎯 Strategic Business Recommendation")

st.info("""
- Prioritize retention strategy for high-risk customers
- Focus on month-to-month contract users
- Improve engagement for medium-risk segments
- Maintain loyalty programs for low-risk customers
""")

# =========================================
# FOOTER
# =========================================

st.divider()

st.caption(
    "M. Wildan Nabila | AI Customer Churn Intelligence System"
)