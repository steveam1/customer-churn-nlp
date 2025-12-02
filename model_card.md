# Model Card for Transformers for Customer Churn Prediction

This model card describes the transformer models that were fine tuned to classify customer reviews as churn risk or no churn risk. The goal of this document is to provide clear and transparent information about the model design, training process, performance characteristics, intended use, limitations, and ethical considerations.

---

## Model Overview

Two transformer models were developed for this project:

1. DistilBERT  
2. RoBERTa

Both models were fine tuned using labeled customer reviews mapped to churn risk categories. DistilBERT is used as a fast and lightweight option, while RoBERTa is used for highest accuracy and strongest overall performance.

---

## Intended Use

The models are intended for:

- predicting customer churn risk from review text  
- ranking customers by churn probability to support proactive retention  
- identifying language patterns that signal customer dissatisfaction  

These models should be used to support human decision making, not replace human judgment.

---

## Training Data

The models were trained on the Yelp Polarity dataset, where:

- positive reviews were mapped to no churn  
- negative reviews were mapped to churn  

The dataset contains:

- 10,000 training samples  
- 2,000 test samples  
- balanced label distribution  

The text was used as provided, with minimal preprocessing so that transformer tokenizers could preserve the raw structure of language.

---

## Model Architecture

Each model consists of:

- a transformer encoder that produces contextual representations  
- a classification head with a single feed forward layer  
- softmax output that produces a churn or no churn probability  

Input text is tokenized using the model specific tokenizer and padded to a maximum length of 128 tokens.

---

## Evaluation Results

Performance on the held out test set:

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|------|--------|
| TF IDF Logistic Regression | 0.9070 | 0.9110 | 0.8999 | 0.9054 | 0.9694 |
| DistilBERT | 0.9155 | 0.9167 | 0.9120 | 0.9143 | 0.9721 |
| RoBERTa | 0.9390 | 0.9313 | 0.9464 | 0.9388 | 0.9867 |

RoBERTa is the top performer and is recommended when accuracy is the priority. DistilBERT is recommended for real time applications requiring lower latency.

---

## Inference Speed

Average per sample inference time:

- Logistic Regression: 0.58 milliseconds  
- DistilBERT: 5.61 milliseconds  
- RoBERTa: 13.52 milliseconds  

These times were measured on GPU with batch size of one.

---

## Explainability

The following interpretability tools were used:

- attention weight visualizations  
- SHAP token level explanations  
- ranking of influential terms  

These tools help users understand how the model arrived at a prediction and which parts of the review text contributed to the decision.

---

## Ethical Considerations and Risks

There are several important ethical considerations:

### 1. Proxy labels  
The dataset uses sentiment as a stand in for true churn behavior. This introduces a modeling limitation and means the output should not be interpreted as a real churn event prediction without further validation.

### 2. Sarcasm and informal language  
Transformers occasionally misinterpret sarcastic comments, mixed sentiment, or slang.

### 3. Domain generalization  
The model may not generalize to domains outside restaurant reviews without further training.

### 4. Fairness  
The dataset does not include demographic information, so demographic fairness cannot be tested. Review length bias was tested and was within acceptable bounds.

---

## Limitations

- The model cannot understand customer context beyond the single review  
- The model does not use structured customer history or behavior data  
- Real world deployment requires additional calibration and testing  
- Predictions should supplement, not replace, human evaluation  

---

## Recommendations for Future Work

- fine tune on a dataset containing real churn labels  
- incorporate additional sources of customer data  
- evaluate performance across different industries  
- add multilingual support  
- include demographic fairness testing when ethically allowed  

---

## Citation

If referencing this model card, please cite:

**Transformers for Customer Churn Prediction  
Model Card Version 1.0**  
