import os
import pickle
import numpy as np
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
DISTILBERT_DIR = "./churn_model/distilbert"
ROBERTA_DIR = "./churn_model/roberta"
BASELINE_MODEL_PATH = "./churn_model/lr_model.pkl"
TFIDF_VECTORIZER_PATH = "./churn_model/tfidf_vectorizer.pkl"
MAX_LENGTH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# LOAD MODELS (CACHED)
# ---------------------------------------------------------
@st.cache_resource
def load_distilbert():
    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_DIR)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


@st.cache_resource
def load_roberta():
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_DIR)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


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
    prob = lr_model.predict_proba(X)[0, 1]
    pred = int(prob >= 0.5)
    return pred, float(prob)


def predict_with_transformer(text: str, model_name="distilbert"):
    if model_name == "distilbert":
        tokenizer, model = load_distilbert()
    else:
        tokenizer, model = load_roberta()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    pred = int(np.argmax(probs))
    prob_churn = float(probs[1])
    return pred, prob_churn


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

st.title("Transformers for Customer Churn Prediction")
st.write(
    """
Paste a customer review and choose a model.
The app predicts how likely the customer is to churn based on the text.
"""
)

# Sidebar
st.sidebar.header("Model settings")

model_choice = st.sidebar.selectbox(
    "Select model",
    ["RoBERTa (best performance)",
     "DistilBERT (fast and accurate)",
     "Baseline TF IDF plus Logistic Regression"],
)

threshold = st.sidebar.slider(
    "Churn risk threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
)

show_details = st.sidebar.checkbox("Show detailed scores", value=True)

st.sidebar.markdown("---")
st.sidebar.write("Device:")
st.sidebar.code(str(DEVICE))

# Input box
example_text = (
    "The food was cold and the service was very slow. "
    "I have come here for years but this visit was really disappointing."
)

review_text = st.text_area(
    "Paste a customer review",
    value=example_text,
    height=160,
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
            if model_choice.startswith("RoBERTa"):
                pred, prob = predict_with_transformer(review_text, "roberta")
                model_label = "RoBERTa"
            elif model_choice.startswith("DistilBERT"):
                pred, prob = predict_with_transformer(review_text, "distilbert")
                model_label = "DistilBERT"
            else:
                pred, prob = predict_with_baseline(review_text)
                model_label = "Baseline TF IDF Logistic Regression"

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
                "The probability shown is the model estimate that the review belongs to the churn class."
            )
            st.code(review_text)


# ---------------------------------------------------------
# EXPLANATION SECTION
# ---------------------------------------------------------
st.markdown("---")
with st.expander("How this works"):
    st.write(
        """
This app uses three models trained on the Yelp Polarity dataset.

1. Baseline  
   Text is converted to TF IDF features and logistic regression predicts churn risk.

2. DistilBERT  
   The review is tokenized and passed through a transformer encoder. The final
   representation goes through a classifier head to estimate churn probability.

3. RoBERTa  
   A larger transformer encoder that provides the highest accuracy.

Class 1 always represents churn risk.  
Adjust the threshold in the sidebar to change how aggressively the model flags churn.
"""
    )
