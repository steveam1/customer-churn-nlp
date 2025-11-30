# Predicting Customer Churn From Reviews Using Transformers

## 📌 Project Overview

Businesses receive thousands of customer reviews every day, but it’s difficult to manually identify which customers might be at risk of leaving. Traditional keyword-based or rule-based methods miss subtle cues like *“might try somewhere else”* versus *“considering alternatives.”*

This project uses Natural Language Processing (NLP) to predict churn risk directly from review text using two models:

1. **TF-IDF + Logistic Regression** (baseline)
2. **Fine-tuned DistilBERT Transformer** (advanced)

Beyond prediction, the project incorporates **interpretability**, including attention visualizations, word importance, and error analysis, to better understand *why* the model makes its predictions.

This model aligns closely with course material—transformer architecture, model fine-tuning, bias analysis, and evaluation metrics—and applies them to a realistic business problem.

---

## 🚀 Key Features

- Two full modeling pipelines (TF-IDF baseline + Transformer)
- High predictive performance (DistilBERT AUC ≈ 0.984)
- Complete evaluation suite: Accuracy, Precision, Recall, F1, AUC
- Interpretability tools: Attention maps, word clouds, calibration curves
- Error analysis: Where models disagree, where they fail
- Saved models + inference functions for deploying predictions

---

## 🧠 Problem Statement

Companies struggle to track churn risk manually across large volumes of reviews. Sentiment alone does not always indicate churn, and subtle signals can easily be missed without contextual modeling.

**Objective:**  
Build a binary classifier that identifies churn risk from review text and outperforms a strong TF-IDF baseline. The model also highlights which phrases indicate churn, helping teams intervene early.

**Target Goals:**
- AUC ≥ 0.85  
- Precision ≥ 70% for top 10% high-risk predictions  
- Inference latency < 100ms  

---

## 🔗 Connection to Course Concepts

This project incorporates several course concepts:

### 🧩 Transformers & Self-Attention
- Fine-tuning DistilBERT for classification  
- Understanding contextual embeddings  
- Visualizing attention weights  

### 📊 Evaluation Metrics
- ROC/AUC  
- Precision, Recall, F1  
- Probability calibration curves  
- Confusion matrix analysis  

### 🛠 Machine Learning Workflow
- Data preprocessing  
- Baseline model development  
- Model comparison  
- Hyperparameter tuning  
- Error and bias analysis  

### ⚖️ Ethical AI
- Bias in language models  
- Reviewer bias and dialect variation  
- Limitations of using sentiment to infer true churn  

---

## 📂 Repository Structure

├── churn_model/
│ ├── distilbert/ # Fine-tuned DistilBERT model + tokenizer
│ ├── lr_model.pkl # Logistic Regression model
│ └── tfidf_vectorizer.pkl # TF-IDF vectorizer vocabulary
│
├── outputs/
│ ├── attention_example_1.png
│ ├── attention_example_2.png
│ ├── attention_example_3.png
│ ├── calibration_curve.png
│ ├── confusion_matrices.png
│ ├── model_comparison.png
│ ├── roc_curves.png
│ ├── word_importance.png
│ └── training_summary.json
│
├── yelp_churn_classification.ipynb
├── requirements.txt
└── README.md

yaml
Copy code

---

## 📊 Dataset

**Dataset:** Yelp Polarity (20,000-sample subset)  
**Source:** https://huggingface.co/datasets/yelp_polarity  
**License:** CC BY 4.0  

### Original Labels  
- 0 → negative (1–2 stars)  
- 1 → positive (4–5 stars)

### Churn Mapping (Project)  
- **Churn (1)** → negative reviews  
- **No Churn (0)** → positive reviews  

**Distribution:** Balanced (≈50/50), stratified across splits.

---

## 📘 Data Card

| Field | Details |
|-------|---------|
| **Dataset** | Yelp Polarity (20k sample) |
| **License** | CC BY 4.0 |
| **Features** | `text`, `label` |
| **Task** | Binary churn classification |
| **Processing** | Tokenization, label remapping, stratified split |
| **Risks** | Reviewer bias, slang/dialect bias, sarcasm, extreme opinions |
| **Limitations** | No metadata, no behavioral churn data, English-only |

---

## 🤖 Model Card — DistilBERT Churn Classifier

### Overview
- Base Model: `distilbert-base-uncased`
- Architecture: 6-layer transformer encoder
- Parameters: ~66M
- Pretrained on large English corpus

### Fine-Tuning Configuration
- Max length: 256 tokens  
- Batch size: 16  
- Epochs: 3  
- Learning rate: 2e-5  
- Warmup: 500 steps  
- Optimizer: AdamW  
- Early stopping enabled  

### Performance (Test Set)

| Metric | Score |
|--------|--------|
| **Accuracy** | 0.9367 |
| **Precision** | 0.9422 |
| **Recall** | 0.9323 |
| **F1 Score** | 0.9372 |
| **AUC** | 0.9838 |

### Intended Use
- Academic projects  
- Research prototyping  
- NLP demonstration and teaching  

### Not Intended For
- Automated decisions without human review  
- Financial, legal, or hiring workflows  
- Commercial deployment without validation  

**License:** Apache 2.0  

---

## 📜 Licenses

- **DistilBERT:** Apache 2.0  
- **Scikit-Learn:** BSD 3-Clause  
- **Yelp Polarity Dataset:** CC BY 4.0  
- **Project Code:** MIT License  

---

## ⚖️ Ethical & Bias Considerations

### Potential Issues
- **Reviewer Bias:** Opinions depend on personal, cultural, or social influences  
- **Language Bias:** Transformers may misinterpret slang or non-standard English  
- **Sentiment vs. Behavior:** Negative sentiment isn’t always churn  
- **Model Bias:** Training data may amplify certain language patterns  

### Mitigation
- Attention maps for transparency  
- Word importance to check model overreliance  
- Error analysis to identify systematic mistakes  
- Clear disclaimers about the model’s limitations  

### Responsible Use Guidance  
This system should **support**, not replace, human judgment. It should not be used to automatically penalize or target customers.

---

## ⚙️ Methodology

### 🔹 1. Baseline: TF-IDF + Logistic Regression
- Vectorizer: 10,000 features  
- n-grams: (1,2)  
- Logistic Regression (balanced class weights)  
- Purpose: Establish a traditional text classification baseline  

### 🔹 2. Transformer: Fine-Tuned DistilBERT
- Tokenization: WordPiece  
- Context-aware embeddings  
- Self-attention mechanism to capture long-range meaning  
- Early stopping + validation monitoring  
- Purpose: Capture nuance that baseline misses  

### Why Transformers?
Transformers understand:
- Negation (e.g., *“not terrible”*)  
- Sarcasm  
- Mixed sentiment  
- Subtle dissatisfaction cues  

---

## 📈 Results Summary

### Test Performance Comparison

| Metric | TF-IDF + LR | DistilBERT | Improvement |
|--------|-------------|------------|-------------|
| Accuracy | 0.9153 | **0.9367** | +2.1% |
| Precision | 0.9122 | **0.9422** | +3.3% |
| Recall | 0.9218 | **0.9323** | +1.1% |
| F1 Score | 0.9169 | **0.9372** | +2.2% |
| AUC | 0.9731 | **0.9838** | +1.1% |

### Key Takeaways
- DistilBERT consistently outperforms the classical baseline  
- Strong improvements in precision and F1  
- Better handling of nuance and ambiguous reviews  
- High baseline limits amount of possible AUC improvement  

---

## 🔍 Interpretability & Analysis

Included tools:
- Confusion matrices  
- ROC curves  
- Calibration curves  
- TF-IDF coefficient importance  
- Attention heatmaps  
- Error breakdown:
  - Baseline wrong / BERT correct  
  - BERT wrong / baseline correct  
  - Both wrong (hard cases)  

Observed patterns:
- Churn cues: “terrible,” “worst,” “never returning,” “rude”  
- Retention cues: “love,” “excellent,” “amazing,” “perfect”  

---

## 🧪 How to Run

### Install Requirements
pip install -r requirements.txt

shell
Copy code

### Run Notebook
jupyter notebook yelp_churn_classification.ipynb

yaml
Copy code

### Google Colab
- Upload notebook  
- Runtime → Change Runtime Type → GPU  
- Run all  

---

## 🔮 Inference

### Baseline Example
```python
predict_churn_baseline("This place was awful. Not coming back.")
DistilBERT Example
python
Copy code
predict_churn_bert("Amazing service! Loved this place.")
🧭 Critical Analysis
Impact
This project shows how NLP can support early churn detection by surfacing dissatisfied customers at scale. It adds transparency through interpretability techniques and connects transformer theory directly to a meaningful real-world use case.

What It Reveals
Transformers capture nuance beyond simple sentiment

Interpretability tools help validate model reasoning

Even with strong baselines, transformers provide measurable gains

Limitations
Yelp reviews don’t reflect true churn behavior

Model trained only on English

No temporal or user-level information

Clean dataset may inflate performance

Next Steps
Add RoBERTa and LoRA/QLoRA

Integrate SHAP explanations

Build Streamlit demo

Add more diverse multi-platform review data

🏁 Conclusion
This project demonstrates how transformer-based NLP models can meaningfully improve churn prediction from text while providing transparent and interpretable explanations. DistilBERT achieves strong performance, outperforming a competitive TF-IDF baseline across all metrics, and aligns with technical and ethical concepts discussed in the course.

Author: Ashley Stevens
Course: LLM Bootcamp
Date: November 2024
