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


def get_word_importance(text: str):
    """Get word importance scores for the review"""
    tfidf_vectorizer, lr_model = load_baseline()
    
    # Get feature names and their coefficients
    feature_names = tfidf_vectorizer.get_feature_names_out()
    coefficients = lr_model.coef_[0]
    
    # Transform the text
    X = tfidf_vectorizer.transform([text])
    
    # Get non-zero features in this review
    feature_indices = X.nonzero()[1]
    
    # Get scores for words actually in this review
    word_scores = {}
    for idx in feature_indices:
        word = feature_names[idx]
        score = coefficients[idx]
        # Only include if the word appears in the original text (handles n-grams)
        if word in text.lower():
            word_scores[word] = score
    
    # Sort by absolute importance
    sorted_words = sorted(word_scores.items(), key=lambda x: abs(x[1]), reverse=True)
    return sorted_words[:10]  # Top 10 most important words


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
Paste a customer review and choose a model. The app predicts how likely the customer is to churn
based on the text.
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
    value=0.20,
    step=0.05,
)

show_details = st.sidebar.checkbox("Show detailed scores", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Device:")
st.sidebar.text("cpu")

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
                "The probability shown is the model estimate that the review "
                "belongs to the churn class."
            )
            
            # Display the review with highlighted important words
            important_words = get_word_importance(review_text)
            
            if important_words:
                st.markdown("#### Key Words Influencing Prediction")
                
                # Separate into churn indicators and loyalty indicators
                churn_words = [(w, s) for w, s in important_words if s > 0]
                no_churn_words = [(w, s) for w, s in important_words if s < 0]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🔴 Churn Indicators:**")
                    if churn_words:
                        for word, score in churn_words[:5]:
                            st.write(f"• **{word}**: {score:.3f}")
                    else:
                        st.write("None detected")
                
                with col2:
                    st.markdown("**🟢 Loyalty Indicators:**")
                    if no_churn_words:
                        for word, score in no_churn_words[:5]:
                            st.write(f"• **{word}**: {abs(score):.3f}")
                    else:
                        st.write("None detected")
            
            st.markdown("---")
            st.code(review_text)


# ---------------------------------------------------------
# EXPLANATION SECTION
# ---------------------------------------------------------
st.markdown("---")
with st.expander("How this works"):
    st.write(
        """
**About the Model:**

This demo uses the baseline TF-IDF + Logistic Regression model from the project.
This model is extremely fast and achieves 96.9% AUC on the test set.

**How It Works:**

1. The text is vectorized using TF-IDF (Term Frequency-Inverse Document Frequency)
2. The logistic regression model predicts churn probability based on learned patterns
3. Words are weighted by their importance - positive weights indicate churn, negative indicate loyalty
4. The model outputs a probability score between 0 and 1

**Full Project:**

The complete project includes transformer models (DistilBERT and RoBERTa) that achieve
even higher accuracy (98.7% AUC), but they require larger model files and GPU for
optimal speed. This demo uses the baseline for lightweight, fast predictions on CPU.

**Performance:**
- Accuracy: 90.7%
- AUC: 96.9%
- Inference: <1ms per prediction
- Top 10% Precision: 99%

See the full project on GitHub for transformer models, comprehensive evaluation,
interpretability analysis, and deployment instructions.
"""
    )

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
    Built with Streamlit • Model: TF-IDF + Logistic Regression • Dataset: Yelp Polarity
    </div>
    """,
    unsafe_allow_html=True,
)
