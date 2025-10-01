# app.py
# Streamlit app to explore the Iris dataset and make predictions.

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import seaborn as sns  # optional for nicer plots

st.set_page_config(page_title="Iris Classifier", layout="wide")

MODEL_FILE = "iris_pipeline.joblib"
DATA_FILE = "iris_dataset.csv"

# ---- Load model and data ----
@st.cache_data
def load_model():
    bundle = joblib.load(MODEL_FILE)
    return bundle["pipeline"], bundle["feature_names"], bundle["target_names"]

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_FILE)

model, feature_cols, target_names = load_model()
data = load_dataset()

# ---- App header ----
st.title("🌸 Iris Data Explorer & Predictor")
st.markdown(
    "Switch between **Explore** and **Predict** in the sidebar. "
    "Explore the dataset or input values to predict the flower species."
)

# ---- Sidebar mode selector ----
st.sidebar.header("Mode")
mode = st.sidebar.radio("Choose mode:", ["Explore", "Predict"])

# Quick peek at the dataset
with st.expander("Dataset Preview"):
    st.dataframe(data.head())

# ---- Exploration mode ----
if mode == "Explore":
    st.subheader("Data Exploration")

    # Histogram
    col_choice = st.selectbox("Column for histogram", feature_cols)
    bins = st.slider("Number of bins", 5, 40, 15)
    fig, ax = plt.subplots()
    ax.hist(data[col_choice], bins=bins, color="skyblue", edgecolor="black")
    ax.set_title(f"Histogram of {col_choice}")
    st.pyplot(fig)

    # Scatter plot
    st.subheader("Scatter Plot")
    x_feat = st.selectbox("X-axis", feature_cols, index=0, key="x")
    y_feat = st.selectbox("Y-axis", feature_cols, index=1, key="y")
    color_opt = st.selectbox("Color by", ["None", "target"])
    fig2, ax2 = plt.subplots()
    if color_opt == "None":
        ax2.scatter(data[x_feat], data[y_feat], alpha=0.7)
    else:
        scatter = ax2.scatter(
            data[x_feat], data[y_feat], c=data[color_opt], cmap="viridis", alpha=0.8
        )
        ax2.legend(*scatter.legend_elements(), title="Class")
    ax2.set_title(f"{y_feat} vs {x_feat}")
    st.pyplot(fig2)

    # Class counts
    st.subheader("Class Counts")
    st.bar_chart(data["target"].value_counts().sort_index())

# ---- Prediction mode ----
else:
    st.subheader("Make a Prediction")
    st.write("Move the sliders to set feature values and click **Predict**.")

    left_col, right_col = st.columns(2)
    user_input = {}

    # Create sliders for each feature based on dataset ranges
    for idx, col in enumerate(feature_cols):
        min_v, max_v = float(data[col].min()), float(data[col].max())
        mean_v = float(data[col].mean())
        with (left_col if idx < 2 else right_col):
            user_input[col] = st.slider(
                label=col,
                min_value=min_v,
                max_value=max_v,
                value=mean_v,
                step=(max_v - min_v) / 100,
                help=f"Range: {min_v:.1f} – {max_v:.1f}"
            )

    if st.button("Predict"):
        input_df = pd.DataFrame([user_input], columns=feature_cols)
        probs = model.predict_proba(input_df)[0]
        pred_class = model.predict(input_df)[0]
        pred_name = target_names[pred_class]

        st.markdown(f"### 🌟 Predicted species: **{pred_name}**")
        prob_table = pd.DataFrame({
            "Species": target_names,
            "Probability": probs
        }).sort_values("Probability", ascending=False)
        st.table(prob_table.style.format({"Probability": "{:.2%}"}))
        st.bar_chart(prob_table.set_index("Species"))

# ---- Sidebar help ----
st.sidebar.markdown("---")
st.sidebar.info(
    "Explore mode: view histograms, scatter plots, and class counts.\n\n"
    "Predict mode: adjust sliders to classify a flower."
)
