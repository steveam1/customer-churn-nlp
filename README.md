# Transformers for Customer Churn Prediction

## Overview

This project predicts whether a customer is at risk of churn based entirely on their written review. Customer feedback contains emotional clues, subtle expressions of frustration, and patterns that relate directly to long term loyalty. Automating the detection of churn risk allows companies to intervene earlier, prioritize the customers most likely to leave, and understand what issues consistently lead to lost loyalty.

I compare three different models:

- A classical natural language approach using TF IDF and logistic regression
- A transformer based DistilBERT model that is fine tuned on the dataset
- A transformer based RoBERTa model that is fine tuned and serves as the strongest performer

Each model is evaluated using accuracy, precision, recall, F1 score, AUC, inference time, and performance on top decile identification. The project also includes interpretability using attention visualizations and SHAP, as well as a review length bias analysis and a complete model card.

A working Streamlit demo is included to show how the model can be used interactively.

---

## 1. Problem Statement

Companies receive large volumes of customer reviews. Reading them manually is slow and inconsistent, and important early warning signals get missed. The central question guiding this project is:

**Can transformer models accurately predict churn risk from raw review text with strong performance and low inference time**

If the answer is yes, this system can support real business use cases such as:

- highlighting customers who need fast attention
- ranking customers by risk level
- identifying common themes behind abandonment
- supporting customer success and retention teams

This project aims to turn unstructured text into actionable customer insight.

---

## 2. Dataset and Preprocessing

The project uses the Yelp Polarity dataset. It includes labeled reviews that are mapped into churn and no churn:

- negative sentiment equals churn risk
- positive sentiment equals no churn risk

The dataset used:

- 10000 training examples
- 2000 testing examples
- balanced classes

Tokenization for transformers uses padding and truncation up to a length of 128 tokens. The text is left mostly raw because transformer tokenizers handle normalization and subword encoding internally.

---

## 3. Baseline Model: TF IDF Logistic Regression

The baseline model provides a strong benchmark. It uses:

- TF IDF features
- one and two gram terms
- a vocabulary capped at 5000 terms
- logistic regression with balanced class weights

### Important churn related words from the baseline model

<p align="center">
  <img src="./outputs/key_churn_phrases.png" width="700">
  <br>
  <em>Key churn related phrases and their importance weights</em>
</p>

This gives an intuitive sense of what a classical model considers risky or safe language.

---

## 4. Transformer Models

### DistilBERT

A smaller and faster variant of BERT. It provides strong accuracy with faster inference. It is well suited for real time scenarios such as live customer support applications.

### RoBERTa

A stronger encoder that produces the best performance in this project. It has a slightly higher inference time but delivers excellent accuracy and AUC.

Both models were fine tuned using:

- three training epochs
- AdamW optimizer
- learning rate of 2e 5
- mixed precision training
- batch size of 32

---

## 5. Model Performance

The models perform as follows:

| Model | Accuracy | Precision | Recall | F1 Score | AUC | Inference Time |
|-------|----------|-----------|--------|----------|-----|----------------|
| TF IDF Logistic Regression | 0.9070 | 0.9110 | 0.8999 | 0.9054 | 0.9694 | 0.58 ms |
| DistilBERT | 0.9155 | 0.9167 | 0.9120 | 0.9143 | 0.9721 | 5.61 ms |
| RoBERTa | 0.9390 | 0.9313 | 0.9464 | 0.9388 | 0.9867 | 13.52 ms |

The strongest performance comes from RoBERTa, which achieves excellent accuracy and robust AUC.

### Performance Visualizations

#### Model Comparison

<p align="center">
  <img src="./outputs/model_performance_comparison.png" width="650">
</p>

#### ROC Curves

<p align="center">
  <img src="./outputs/roc_curves.png" width="650">
</p>

#### Calibration Curves

<p align="center">
  <img src="./outputs/calibration_curves.png" width="650">
</p>

#### Inference Latency Comparison

<p align="center">
  <img src="./outputs/inference_latency_comparison.png" width="650">
</p>

---

## 6. Confusion Matrices

Confusion matrices show the balance between correct and incorrect predictions.

<p align="center">
  <img src="./outputs/confusion_matrices.png" width="700">
  <br>
  <em>Confusion matrices for all three models</em>
</p>

These demonstrate that the models do not show concerning imbalance or systematic bias in their predictions.

---

## 7. Top Ten Percent Precision

Many business applications focus on the top group of customers who show the highest risk. This analysis looks at the precision for the top ten percent of predicted churn probabilities.

<p align="center">
  <img src="./outputs/top10_precision.png" width="600">
</p>

Results:

- Baseline: 99 percent
- DistilBERT: 100 percent
- RoBERTa: 100 percent

This shows that all models are excellent at ranking the highest risk customers.

---

## 8. Interpretability

Interpretability supports trust in model predictions and helps show which words or phrases influence the classification.

### Attention Visualizations

<p align="center">
  <img src="./outputs/attention_example_1.png" width="650">
  <br>
  <em>Example of attention focus for a high risk review</em>
</p>

<p align="center">
  <img src="./outputs/attention_example_2.png" width="650">
  <br>
  <em>Example of attention focus for a no risk review</em>
</p>

These visualizations highlight the phrases that the transformer finds important.

### SHAP Explanations

<p align="center">
  <img src="./outputs/shap_example.png" width="650">
  <br>
  <em>SHAP values showing token contributions toward churn risk</em>
</p>

SHAP provides a clear explanation of how each word contributes to a prediction.

---

## 9. Bias Analysis

Bias is evaluated by review length. This checks whether the model performs differently based on how long a review is.

<p align="center">
  <img src="./outputs/bias_analysis.png" width="650">
  <br>
  <em>Review length accuracy analysis</em>
</p>

The maximum difference across groups is eight percent. This is within the safe range defined for this project, so no major bias concern is identified.

---

## 10. Misclassification Review

A detailed review of incorrect predictions reveals the following common patterns:

- mixed emotion reviews
- reviews with long stories that bury sentiment
- short sarcastic comments
- ambiguous sentiment phrases

These insights help identify future directions for improvement.

---

## 11. Streamlit Demo

A working Streamlit demo is included at:

```
streamlit_app/app.py
```

To run the demo:

```bash
pip install -r requirements.txt
streamlit run streamlit_app/app.py
```

This launches an interactive interface where you can paste a review and choose a model. The demo outputs the churn probability and a classification decision.

---

## 12. Model Card

### Model Name

DistilBERT and RoBERTa for churn classification

### Intended Use

Predicting churn risk from customer review text

### Input

Raw customer review text up to 128 tokens

### Output

- Binary churn classification and probability score

### Performance Summary

- AUC up to 0.9867
- F1 score up to 0.9388
- Top ten percent precision of 100 percent for transformer models

### Ethical Considerations

The dataset does not contain demographic attributes so demographic bias cannot be evaluated. The sentiment based labels act only as a proxy for true churn behavior. The model may misinterpret sarcasm, slang, or highly ambiguous text.

### Limitations

The model is trained on sentiment mapped churn labels, not real churn events. It may not generalize to other domains without further tuning.

---

## 13. Data Card

### Dataset

Yelp Polarity Dataset

### Labeling

- Positive reviews mapped to no churn
- Negative reviews mapped to churn

### Distribution

Balanced classes in training and test sets

### Notes

Sentiment is only a proxy for churn. No demographic information is available in the dataset.

---

## 14. Critical Analysis

Transformers significantly outperform the classical model while maintaining low inference times. RoBERTa performs particularly well and provides strong ranking ability in top decile analysis.

Interpretability helps clarify how predictions are made, and bias analysis shows consistent performance across review length categories. The primary limitation is that sentiment labels are not true churn labels. Real world deployment would require training on actual churn outcomes and more diverse customer behavior data.

---

## 15. Next Steps

Future improvements include:

- training on real churn behavior
- incorporating structured customer data
- adding time based features
- applying topic modeling to identify common dissatisfaction themes
- evaluating multilingual reviews
- building an end to end dashboard including volume trends, reasons for churn, and customer attributes

---

## 16. Repository Structure

```
project_root/
│
├── churn_model/
│   ├── lr_model.pkl
│   ├── tfidf_vectorizer.pkl
│
├── data/
│   ├── model_comparison.csv
│   ├── training_summary.json
│
├── outputs/
│   ├── calibration_curves.png
│   ├── confusion_matrices.png
│   ├── inference_latency_comparison.png
│   ├── key_churn_phrases.png
│   ├── model_performance_comparison.png
│   ├── roc_curves.png
│   ├── top10_precision.png
│   ├── bias_analysis.png
│   ├── attention_example_1.png
│   ├── attention_example_2.png
│   ├── shap_example.png
│
├── streamlit_app/
│   ├── app.py
│
├── requirements.txt
├── Churn_Prediction.ipynb
└── README.md
```

---

## Conclusion

This project demonstrates how transformer models turn raw customer text into accurate and meaningful churn predictions. With strong performance, thoughtful interpretability, a working demo, and complete documentation, this project shows both technical skill and practical business relevance.


