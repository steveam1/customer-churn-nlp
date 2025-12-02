import os
import pickle
import numpy as np
import torch
import streamlit as st

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
BASELINE_MODEL_PATH = "./churn_model/lr_model.pkl"
TFIDF_VECTORIZER_PATH = "./churn_model/tfidf_vectorizer.pkl"


# ---------------------------------------------------------
# LOAD BASELINE MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_baseline():
    with open(BASELINE_MODEL_PATH, "rb") as f:
        lr_model = pickle.load(f)
    with open(TFIDF_VECTORIZER_PATH, "rb") as f:
        tfidf_vectorizer = pickle.load(f)
    return tfidf_vectorizer, lr_model


# ---------------------------------------------------------
# PREDICTION HELPERS
# ---------------------------------------------------------
def predict_with_baseline(text: str):
    tfidf_vectorizer, lr_model = load_baseline()
    X = tfidf_vectorizer.transform([text])
    prob = lr_model.predict_proba(X)[0, 1]  # probability of churn class
    pred = int(prob >= 0.5)
    return pred, float(prob)


def format_prediction(pred: int, prob: float, threshold: float):
    label = "Churn risk" if prob >= threshold else "No churn risk"
    emoji = "⚠️" if label == "Churn risk" else "✅"
    return f"{emoji} {label}", f"{prob * 100:.2f}%"


# ---------------------------------------------------------
# STREAMLIT APP UI
# ---------------------------------------------------------
st.set_page_config(
    page_title="Transformers for Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
)

st.title("Customer Churn Prediction from Review Text")
st.write(
    """
Paste a customer review and the app will estimate how likely the customer is
to churn based on the content of the review.

This demo uses the baseline TF IDF plus Logistic Regression model 
for lightweight and fast predictions.
"""
)

# Sidebar
st.sidebar.header("Model settings")

model_choice = st.sidebar.selectbox(
    "Select model",
    ["Baseline TF IDF plus Logistic Regression"],
)

threshold = st.sidebar.slider(
    "Churn risk threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
)

show_details = st.sidebar.checkbox("Show detailed scores", value=True)

# Example review
example_text = (
    "The food was cold and the service was very slow. "
    "I have come here for years but this visit was really disappointing."
)

review_text = st.text_area(
    "Paste a customer review",
    value=example_text,
    height=150,
)

predict_button = st.button("Predict churn risk")


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
if predict_button:
    if not review_text.strip():
        st.warning("Please enter a review first.")
    else:
        with st.spinner("Running model..."):
            pred, prob = predict_with_baseline(review_text)
            model_label = "Baseline TF IDF plus Logistic Regression"

        label_text, prob_text = format_prediction(pred, prob, threshold)

        st.subheader("Prediction")
        st.markdown(f"**Model used:** {model_label}")
        st.markdown(f"**Result:** {label_text}")
        st.markdown(f"**Predicted churn probability:** {prob_text}")
        st.markdown(f"**Decision threshold:** {threshold:.2f}")

        if show_details:
            st.markdown("### Details")
            st.write(
                "Class 1 represents churn risk. "
                "The probability shown is the model estimate that the review "
                "belongs to the churn class."
            )
            st.code(review_text)


# ---------------------------------------------------------
# EXPLANATION SECTION
# ---------------------------------------------------------
st.markdown("---")
with st.expander("How this works"):
    st.write(
        """
This demo uses the baseline TF IDF plus Logistic Regression model from the project.
This model is extremely fast and still performs well with an AUC of 0.969.

The full project includes transformer models such as DistilBERT and RoBERTa,
but those require much larger model files and are not included in this demo
to keep the app lightweight.

The baseline model vectorizes the text using TF IDF and predicts churn risk
based on patterns learned from the training data.
"""
    )

