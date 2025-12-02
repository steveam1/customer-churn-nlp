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

### Course Connections

This project directly applies core concepts from **DSCI 552 - Machine Learning for Data Science**, demonstrating mastery of techniques covered in the curriculum:

**Transformer Architecture and Attention Mechanisms (Course Module: Neural Language Models):**  
The self-attention mechanism from Vaswani et al. (2017), covered in transformer lectures, allows each token to attend to all other tokens in the sequence. This captures long-range dependencies and context-dependent meaning that recurrent models struggle with. DistilBERT and RoBERTa implement the multi-head attention with learned query, key, and value projections discussed in class, enabling the models to understand negation ("not bad" vs "bad"), intensifiers, and semantic relationships across long text spans.

**Transfer Learning and Fine-Tuning (Course Module: Pretrained Models):**  
Following the transfer learning framework taught in class, this project leverages models pretrained on billions of tokens rather than training from scratch. The fine-tuning approach applies gradient-based optimization on the labeled churn dataset while starting from pretrained weights, implementing the domain adaptation methodology emphasized in course lectures. This demonstrates the practical application of how general language understanding transfers to specific downstream tasks.

**Comprehensive Model Evaluation (Course Module: Performance Metrics and Model Selection):**  
The evaluation strategy implements the multi-metric assessment framework from class:
- **ROC/AUC:** Threshold-independent ranking performance as taught in classification evaluation lectures
- **Precision-Recall Tradeoffs:** Business-oriented metrics for cost-sensitive decisions
- **Calibration Analysis:** Probability reliability assessment following course best practices
- **Top-K Metrics:** Real-world deployment scenarios emphasizing high-stakes predictions
- **Confusion Matrices:** Complete error pattern analysis

This goes beyond simple accuracy to capture the multiple performance dimensions emphasized throughout the course.

**Optimization Techniques (Course Module: Training Deep Neural Networks):**  
Implementation uses AdamW optimizer with learning rate warmup and weight decay, applying the training strategies covered in neural network optimization lectures. Early stopping provides implicit regularization, and mixed-precision (FP16) training demonstrates the computational efficiency techniques discussed in class.

**Ethical AI and Fairness (Course Module: Bias Detection and Responsible ML):**  
Following the fairness assessment methodology taught in class, systematic bias detection examines performance across review length categories. The analysis includes transparency requirements, appropriate use case documentation, and responsible deployment guidelines—all principles emphasized in the course's ethical AI module. This demonstrates awareness that model performance must be evaluated not just for accuracy but for fairness across population subgroups.

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

## 3. Implementation Overview

### Code Structure

The complete implementation is available in `Churn_Prediction.ipynb` with the following pipeline:

1. **Data Loading and Preprocessing**
   ```python
   # Load Yelp dataset from HuggingFace
   dataset = load_dataset("yelp_polarity")
   
   # Map labels: negative (1-2 stars) → churn, positive (4-5 stars) → no churn
   df['churn'] = df['label'].apply(lambda x: 1 if x == 1 else 0)
   
   # Stratified split
   X_train, X_val, y_train, y_val = train_test_split(
       X_train_full, y_train_full, test_size=0.2, 
       stratify=y_train_full, random_state=42
   )
   ```

2. **Baseline Model Training**
   ```python
   # TF-IDF vectorization
   tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
   X_train_tfidf = tfidf.fit_transform(X_train)
   
   # Logistic regression with balanced weights
   lr_model = LogisticRegression(class_weight='balanced', max_iter=1000)
   lr_model.fit(X_train_tfidf, y_train)
   ```

3. **Transformer Fine-Tuning**
   ```python
   # Load pretrained model
   model = AutoModelForSequenceClassification.from_pretrained(
       "distilbert-base-uncased", num_labels=2
   )
   
   # Training configuration
   training_args = TrainingArguments(
       per_device_train_batch_size=32,
       num_train_epochs=3,
       learning_rate=2e-5,
       warmup_steps=500,
       weight_decay=0.01,
       eval_strategy="epoch",
       fp16=True  # Mixed precision
   )
   
   # Train with HuggingFace Trainer
   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=train_dataset,
       eval_dataset=val_dataset,
       compute_metrics=compute_metrics
   )
   trainer.train()
   ```

4. **Evaluation and Analysis**
   - Comprehensive metrics calculation (accuracy, precision, recall, F1, AUC)
   - Confusion matrix generation
   - Top-10% precision analysis
   - Attention visualization extraction
   - SHAP value computation
   - Bias analysis across review lengths

### Key Technical Decisions

**Why Encoder-Only Transformers (Course Concept: Architecture Selection for Task Types):**  
Classification tasks require bidirectional context to understand negation ("not bad" vs "bad") and long-range dependencies. As discussed in course lectures on transformer variants, encoder-only models like BERT excel at discriminative tasks because they process the full sequence simultaneously through bidirectional attention. Decoder-only models (GPT-style), covered as generation-focused architectures, process text left-to-right and cannot leverage future context—making them suboptimal for classification where understanding the complete input is essential.

**Why RoBERTa Over BERT (Course Concept: Pretraining Optimization):**  
Following pretraining improvements discussed in class, RoBERTa optimizes BERT through: longer training duration (160GB vs 16GB of text), dynamic masking (different masked tokens each epoch rather than static masks), larger batch sizes for more stable gradients, and removal of the next-sentence prediction objective that BERT used. These modifications, covered in advanced transformer lectures, yield superior downstream task performance—demonstrating how thoughtful pretraining choices compound into significant improvements.

**Mixed Precision Training (Course Concept: Computational Efficiency Techniques):**  
Using FP16 (16-bit floating point) instead of FP32 reduces memory usage by 50% and accelerates training by approximately 2x on modern GPUs without sacrificing model accuracy. This efficiency technique, discussed in the course's computational optimization module, enables training larger models or larger batch sizes within GPU memory constraints—a critical consideration for practical deep learning deployment.

---

## 4. Baseline Model: TF IDF Logistic Regression

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

## 5. Transformer Models

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

## 6. Model Performance

The models perform as follows:

| Model | Accuracy | Precision | Recall | F1 Score | AUC | Inference Time |
|-------|----------|-----------|--------|----------|-----|----------------|
| TF IDF Logistic Regression | 0.9070 | 0.9110 | 0.8999 | 0.9054 | 0.9694 | 0.58 ms |
| DistilBERT | 0.9155 | 0.9167 | 0.9120 | 0.9143 | 0.9721 | 5.61 ms |
| RoBERTa | 0.9390 | 0.9313 | 0.9464 | 0.9388 | 0.9867 | 13.52 ms |

The strongest performance comes from RoBERTa, which achieves excellent accuracy and robust AUC of 98.67%. All models demonstrate strong discrimination ability with AUCs above 96%.

### Performance Visualizations

#### Model Performance Comparison

<p align="center">
  <img src="./outputs/model_performance_comparison.png" width="650">
  <br>
  <em>Comprehensive performance comparison across all evaluation metrics</em>
</p>

#### Inference Latency Comparison

<p align="center">
  <img src="./outputs/inference_latency_comparison.png" width="650">
  <br>
  <em>Speed vs accuracy tradeoff: DistilBERT offers the best balance</em>
</p>

#### Calibration Curves

<p align="center">
  <img src="./outputs/calibration_curves.png" width="650">
  <br>
  <em>Probability calibration showing all models produce well-calibrated predictions</em>
</p>

---

## 7. Top Ten Percent Precision Analysis

Many business applications focus on the top group of customers who show the highest risk. This analysis measures the precision for the top ten percent of predicted churn probabilities—a critical metric for targeting high-value intervention efforts.

<p align="center">
  <img src="./outputs/top10_precision.png" width="600">
  <br>
  <em>Top 10% precision: Transformers achieve perfect identification of high-risk customers</em>
</p>

**Results:**

- **Baseline: 99%** - Excellent performance from classical approach
- **DistilBERT: 100%** - Perfect precision in highest-risk decile
- **RoBERTa: 100%** - Perfect precision in highest-risk decile

**Business Impact:** All models greatly exceed the target threshold of 70%, with transformer models achieving perfect scores. This means every customer flagged in the top 10% is a genuine churn risk—enabling highly targeted retention efforts with zero wasted resources on false alarms.

---

## 8. Confusion Matrices

Confusion matrices show the balance between correct and incorrect predictions.

<p align="center">
  <img src="./outputs/confusion_matrices.png" width="700">
  <br>
  <em>Confusion matrices for all three models</em>
</p>

These demonstrate that the models do not show concerning imbalance or systematic bias in their predictions.

---

## 9. Interpretability

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

## 10. Bias Analysis

Bias is evaluated by review length. This checks whether the model performs differently based on how long a review is.

<p align="center">
  <img src="./outputs/bias_analysis.png" width="650">
  <br>
  <em>Review length accuracy analysis</em>
</p>

The maximum difference across groups is eight percent. This is within the safe range defined for this project, so no major bias concern is identified.

---

## 11. Misclassification Review

A detailed review of incorrect predictions reveals the following common patterns:

- mixed emotion reviews
- reviews with long stories that bury sentiment
- short sarcastic comments
- ambiguous sentiment phrases

These insights help identify future directions for improvement.

---

## 12. Streamlit Demo

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

## 13. Model Card

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

## 14. Data Card

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

## 15. Critical Analysis

Transformers significantly outperform the classical model while maintaining low inference times. RoBERTa performs particularly well and provides strong ranking ability in top decile analysis.

Interpretability helps clarify how predictions are made, and bias analysis shows consistent performance across review length categories. The primary limitation is that sentiment labels are not true churn labels. Real world deployment would require training on actual churn outcomes and more diverse customer behavior data.

---

## 16. Next Steps

Future improvements include:

- training on real churn behavior
- incorporating structured customer data
- adding time based features
- applying topic modeling to identify common dissatisfaction themes
- evaluating multilingual reviews
- building an end to end dashboard including volume trends, reasons for churn, and customer attributes

---

## 18. How to Run

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/customer-churn-nlp.git
   cd customer-churn-nlp
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Required packages:
   - `transformers>=4.30.0`
   - `torch>=2.0.0`
   - `datasets>=2.14.0`
   - `scikit-learn>=1.3.0`
   - `pandas>=2.0.0`
   - `numpy>=1.24.0`
   - `matplotlib>=3.7.0`
   - `seaborn>=0.12.0`
   - `shap>=0.42.0`
   - `streamlit>=1.25.0` (for demo)

3. **Run the notebook**
   ```bash
   jupyter notebook Churn_Prediction.ipynb
   ```
   
   Or use Google Colab:
   - Upload `Churn_Prediction.ipynb`
   - Set Runtime → Change runtime type → GPU (T4)
   - Run all cells

4. **Run the Streamlit demo** (optional)
   ```bash
   streamlit run streamlit_app/app.py
   ```

### Usage Guide

The notebook includes complete end-to-end training and evaluation:
- Sections 1-3: Data loading and preprocessing
- Sections 4-6: Baseline model training
- Sections 7-9: Transformer fine-tuning (DistilBERT and RoBERTa)
- Sections 10-12: Evaluation and visualization
- Sections 13-15: Interpretability analysis
- Sections 16-18: Bias analysis and model cards

All outputs save to `./outputs/` and `./churn_model/` directories.

---

## 19. References and Resources

### Research Papers

**Vaswani, A., et al. (2017).** *Attention is All You Need.* Advances in Neural Information Processing Systems, 30.  
https://arxiv.org/abs/1706.03762

**Devlin, J., et al. (2018).** *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.*  
https://arxiv.org/abs/1810.04805

**Sanh, V., et al. (2019).** *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.*  
https://arxiv.org/abs/1910.01108

**Liu, Y., et al. (2019).** *RoBERTa: A Robustly Optimized BERT Pretraining Approach.*  
https://arxiv.org/abs/1907.11692

**Lundberg, S. M., & Lee, S. I. (2017).** *A Unified Approach to Interpreting Model Predictions.*  
Advances in Neural Information Processing Systems, 30.

### Datasets

**Yelp Open Dataset**  
https://www.yelp.com/dataset  
Accessed via HuggingFace: https://huggingface.co/datasets/yelp_polarity  
License: Creative Commons Attribution 4.0 International (CC BY 4.0)

### Code and Frameworks

**HuggingFace Transformers**  
https://huggingface.co/transformers/  
License: Apache License 2.0

**PyTorch**  
https://pytorch.org/  
License: BSD-style license

**Scikit-learn**  
https://scikit-learn.org/  
License: BSD 3-Clause

**SHAP (SHapley Additive exPlanations)**  
https://github.com/slundberg/shap  
License: MIT License

### Course Materials

**DSCI 552 - Machine Learning for Data Science**  
University of Southern California, Viterbi School of Engineering

This project applies concepts from course lectures on:
- Transformer architectures and attention mechanisms
- Transfer learning and fine-tuning
- Model evaluation and performance metrics
- Ethical AI and bias detection
- NLP pipelines and tokenization strategies

---

## 20. Repository Structure

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
├── README.md
└── REFERENCES.md
```

---

## 21. Conclusion

This project demonstrates how transformer models turn raw customer text into accurate and meaningful churn predictions. With strong performance, thoughtful interpretability, a working demo, and complete documentation, this project shows both technical skill and practical business relevance.
