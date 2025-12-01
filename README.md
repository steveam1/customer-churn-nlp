# Predicting Customer Churn from Reviews Using Transformer Models

## 1. Overview

Businesses rely heavily on understanding customer satisfaction, yet it is difficult to manually review large volumes of customer feedback and identify who is at risk of leaving. Traditional keyword systems and simple sentiment rules often miss subtle cues in language, such as negation, sarcasm, or mixed sentiment.

This project builds and evaluates machine learning models that classify whether a customer is likely to churn based solely on the text of their review. I compare a classical TF-IDF baseline with two modern transformer-based language models, DistilBERT and RoBERTa.

The results show that transformer models provide clear performance improvements, with RoBERTa achieving an AUC of 0.987 and reaching perfect precision on the top ten percent of high-risk predictions. These findings demonstrate both the practical value of advanced NLP models and their connection to core concepts from the course, such as self-attention, transfer learning, and model interpretability.

---

## 2. Problem Statement and Business Context

Customer churn is expensive. Retaining a customer often costs far less than acquiring a new one, so organizations benefit from early identification of customers who are dissatisfied or thinking about switching providers.

Companies collect thousands of reviews, but reviewing them manually is slow and inconsistent. Existing sentiment systems also fall short because:

- Sentiment does not always equal churn
- Keywords cannot capture the meaning of a phrase in context
- Sarcasm, intensifiers, and negations drastically change meaning
- Important business cues come from subtle language (for example, "not sure I will return" or "thinking about trying somewhere else")

### Research Objectives

The project aims to build a text-based churn classifier that:

- Achieves an AUC of at least 0.85
- Reaches at least 70 percent precision on the top ten percent of high-risk predictions
- Meets real-time inference requirements with latency below 100 milliseconds

The broader goal is to show how transformer-based NLP can support customer retention strategies with interpretable, efficient, and well-calibrated predictions.

---

## 3. Connection to Course Concepts

This project directly applies many of the topics covered in class, including:

### Transformers and Self-Attention

Transformers replace recurrence with self-attention. Each token can attend to every other token, allowing the model to capture long-range dependencies and nuanced semantic patterns. DistilBERT uses knowledge distillation to compress BERT, while RoBERTa improves pretraining for stronger performance.

These ideas connect directly to course material on:

- Query, key, and value projections
- Multi-head self-attention
- Positional encodings
- Model scaling and efficiency
- Transfer learning and fine-tuning

### Formal Algorithms Concepts

The project reinforces algorithmic ideas such as:

- Complexity differences between classical models and transformers
- Optimization through gradient descent
- Regularization in logistic regression
- Convergence and generalization

### NLP Pipeline Concepts

- Tokenization
- Vectorization (TF-IDF)
- Contextual embeddings
- Classification head design
- Evaluation metrics and calibration

The final result ties together the entire course: algorithms, NLP, implementation, training, and real-world application.

---

## 4. Dataset and Label Construction

### Data Source

I use a ten thousand sample subset of the Yelp Polarity dataset from HuggingFace.
The dataset includes written reviews with polarity labels.

- Negative reviews (one to two stars) map to churn
- Positive reviews (four to five stars) map to no churn
- Neutral three-star reviews are removed to form a clear binary problem

The final dataset is balanced, with about fifty percent churn reviews.

### Data Splitting

The sample is divided with stratified sampling:

- Training: 80 percent (8,000 samples)
- Validation: 10 percent (2,000 samples)
- Test: 10 percent (2,000 samples)

A fixed seed is used for reproducibility.

### Data Card

- **Features:** review text, churn label
- **Language:** English
- **Risks:** reviewer bias, slang or dialect variation, ambiguous text
- **Limitations:** no metadata or behavioral churn information

---

## 5. Methodology

### Classical Baseline: TF-IDF with Logistic Regression

This model vectorizes text using unigrams and bigrams and trains a logistic regression classifier with balanced class weights. This provides a strong baseline for comparison and allows direct interpretation of word importance.

### Transformer Models

Both transformer models are fine-tuned end to end on the churn classification task.

#### DistilBERT

- Six-layer encoder
- Around 66 million parameters
- Designed for efficiency and faster inference

#### RoBERTa

- Twelve layers
- Around 125 million parameters
- Trained with improved pretraining objectives and more data
- Expected to capture more nuanced language patterns

### Training Configuration

Training for both models uses:

- Max sequence length of 128 tokens
- Batch size of 32
- Learning rate of 2e-5 with warmup
- AdamW optimization
- Early stopping based on validation performance

---

## 6. Evaluation and Results

### Performance Summary

| Metric | TF-IDF + Logistic Regression | DistilBERT | RoBERTa |
|--------|------------------------------|------------|---------|
| Accuracy | 90.7% | 91.6% | 93.9% |
| Precision | 91.1% | 91.7% | 93.1% |
| Recall | 90.0% | 91.2% | 94.6% |
| F1 Score | 90.5% | 91.4% | 93.9% |
| AUC | 96.9% | 97.2% | 98.7% |

RoBERTa achieves the strongest performance overall, especially in recall and AUC, which indicate stronger ability to identify churn.

### Model Performance Comparison

![Model Performance Comparison](outputs/model_performance_comparison.png)

### ROC Curves

![ROC Curves](outputs/roc_curves.png)

### Confusion Matrices

![Confusion Matrices](outputs/confusion_matrices.png)

### Top Ten Percent Precision

DistilBERT and RoBERTa both reach 100 percent precision in the top ten percent of predicted churn risk, which far exceeds the seventy percent target.

![Top 10% Precision](outputs/top10_precision.png)

### Calibration Curves

Both transformer models produce well-calibrated probabilities, which is important for making threshold-based business decisions.

![Calibration Curves](outputs/calibration_curves.png)

### Inference Latency

All models meet the one hundred millisecond latency requirement, and DistilBERT provides a good balance between accuracy and running time.

![Inference Latency](outputs/inference_latency_comparison.png)

---

## 7. Interpretability

### TF-IDF Word Importance

These coefficients highlight which words drive predictions. Negations and extreme negative adjectives strongly indicate churn, while emotional and superlative terms indicate loyalty.

![Key Churn Phrases](outputs/key_churn_phrases.png)

### Attention Patterns

Attention visualizations reveal how the models use context. Examples show:

- Strong attention between negation words and the sentiment they reverse
- High weights linking intensifiers with their modified words
- Cross-sentence and long-span dependencies in longer reviews

These patterns help validate that the model is focusing on meaningful linguistic cues.

---

## 8. Critical Analysis

### What the Results Reveal

- Transformers clearly outperform classical models on nuanced language
- The TF-IDF model is surprisingly strong, which suggests sentiment is a strong proxy for churn
- RoBERTa's higher capacity allows it to capture more subtle patterns
- The models show no major length bias, although very long reviews still pose challenges
- High top decile precision indicates the models are especially reliable when used for targeted interventions

### Limitations

- Labels are based on sentiment rather than observed churn behavior
- Yelp reviewers are not representative of all customer groups
- Only English text is included
- No customer metadata or business context is used
- Real-world deployment requires ongoing monitoring and fairness audits

### Ethical Considerations

Models may behave differently across dialects, writing styles, or demographic groups. For responsible use:

- Predictions should support, not replace, human decision making
- Businesses should avoid using the model for punitive actions
- Regular audits should check for unintentional bias
- Probabilities should be presented with uncertainty awareness

---

## 9. How to Run the Project

1. Clone the repository
2. Install dependencies with `pip install -r requirements.txt`
3. Open the notebook `Churn_Prediction.ipynb`
4. Select GPU runtime if using Colab
5. Run all cells to reproduce results
6. Saved outputs will appear in the `outputs/` folder

---

## 10. Repository Structure

```
churn_model/
    distilbert/                # Fine-tuned DistilBERT model
    roberta/                   # Fine-tuned RoBERTa model
    lr_model.pkl               # TF-IDF logistic regression model
    tfidf_vectorizer.pkl       # Vocabulary and vectorizer

outputs/
    roc_curves.png
    confusion_matrices.png
    model_performance_comparison.png
    calibration_curves.png
    top10_precision.png
    key_churn_phrases.png
    inference_latency_comparison.png

data/
    model_comparison.csv
    training_summary.json

Churn_Prediction.ipynb
README.md
REFERENCES.md
requirements.txt
```

---

## 11. Conclusion

This project demonstrates that transformer models offer meaningful improvements in churn prediction from review text. RoBERTa delivers the strongest performance, achieving an AUC of 0.987 and perfect precision for the highest-risk customers.

The analysis ties together key course concepts, including self-attention, transfer learning, algorithmic complexity, and evaluation methods. It also emphasizes interpretability, fairness, and practical deployment considerations.

Although the dataset presents limitations, the results show clear value in applying transformer-based NLP to real business challenges. Future work could incorporate verified churn labels, additional data sources, multilingual text, or more advanced explainability tools.

---

## References

See [REFERENCES.md](REFERENCES.md) for complete citations and acknowledgments.

---

**Author:** Ashley Stevens  
**Course:** DSCI 552 - Machine Learning for Data Science  
**Institution:** University of Southern California  
**Date:** December 2024


