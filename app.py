import os
import joblib
import pandas as pd
import streamlit as st

from config import MODEL_PATH, DAGSHUB_USERNAME, DAGSHUB_REPO_NAME

st.set_page_config(
    page_title="Student Marks Predictor",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Student Marks Predictor")
st.markdown(
    "Predict student exam marks based on daily study hours using a trained "
    "**Linear Regression** model tracked with **MLflow & DagsHub**."
)

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found. Run `python train.py` first "
        "and include `artifacts/model.pkl` in the repository for deployment."
    )
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Sidebar with project info
with st.sidebar:
    st.header("ℹ️ Project Info")
    st.markdown(
        f"""
        - **Repository**: [{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}](https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME})
        - **MLflow Tracking**: [DagsHub MLflow](https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow)
        - **Model Type**: Linear Regression
        - **Framework**: Scikit-Learn & Streamlit
        """
    )
    st.divider()
    st.caption("Simple MLOps Pipeline Demonstration")

col1, col2 = st.columns([2, 1])

with col1:
    hours = st.slider(
        "Study Hours per Day:",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5,
        help="Select the number of hours studied per day"
    )

with col2:
    st.write("")
    st.write("")
    predict_clicked = st.button("Predict Marks", type="primary", use_container_width=True)

sample = pd.DataFrame({"hours": [hours]})
raw_prediction = float(model.predict(sample)[0])
prediction = max(0.0, min(100.0, raw_prediction))

# Prediction Display
st.subheader("Prediction Result")
metric_col1, metric_col2, metric_col3 = st.columns(3)
with metric_col1:
    st.metric(label="Predicted Score", value=f"{prediction:.2f} / 100")
with metric_col2:
    status = "🌟 Distinction" if prediction >= 75 else ("👍 First Class" if prediction >= 60 else ("✅ Pass" if prediction >= 40 else "⚠️ Needs Improvement"))
    st.metric(label="Performance Tier", value=status)
with metric_col3:
    st.metric(label="Study Hours", value=f"{hours} hrs/day")

st.progress(min(1.0, max(0.0, prediction / 100.0)))

# Visual Representation
st.subheader("📈 Regression Trend")
trend_hours = list(range(0, 13))
chart_data = pd.DataFrame({
    "Study Hours": trend_hours,
    "Predicted Marks": [max(0.0, min(100.0, float(model.predict(pd.DataFrame({'hours': [h]}))[0]))) for h in trend_hours]
})
st.line_chart(chart_data.set_index("Study Hours"), color="#2563EB")

with st.expander("🔍 View Model Details"):
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    st.markdown(f"**Regression Formula:** `Marks = {slope:.2f} × Hours + ({intercept:.2f})`")
    st.markdown(
        f"- **Slope (Weight):** Each additional study hour increases the predicted score by approximately **{slope:.2f} marks**.\n"
        f"- **Intercept (Base):** Expected score with 0 study hours is **{intercept:.2f} marks**."
    )

st.caption(
    "Educational demonstration using synthetic data. Experiment tracked with MLflow on DagsHub."
)
