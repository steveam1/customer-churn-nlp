 Predicting Customer Churn from Customer Reviews Using Transformer Models

This project builds and compares three models for predicting customer churn from text reviews. I use a classical baseline (TF IDF with Logistic Regression) and two modern Transformer models (DistilBERT and RoBERTa). The goal is to understand how well contextual language models can identify linguistic signals of dissatisfaction, and whether they meaningfully outperform traditional NLP approaches.

The final RoBERTa model reaches an AUC of 0.987 and perfect precision in the top ten percent of highest-risk predictions. This shows that Transformers can capture subtle cues in customer language that older models often miss.

---

## 1. Problem Statement

Customer churn is expensive, and companies want to identify customers who are at risk based on what they write in reviews. Traditional sentiment methods often miss important context, such as negation, tone, or mixed sentiment. Transformers can model these signals using self attention, which allows each word to consider every other word in the sentence.

The core objective of the project is to build a model that:

- Achieves strong ranking performance (AUC ≥ 0.85)
- Maintains at least 70 percent precision in the highest-risk group
- Produces predictions with low latency so it can run in real time

---

## 2. Course Integration and Methodology

This project directly applies topics from the course:

- Transformer architecture and self attention (Vaswani et al., 2017)
- Fine tuning pretrained models such as DistilBERT and RoBERTa
- Parameter efficiency and practical training considerations
- Evaluation frameworks including ROC curves, calibration, and bias checks
- Model cards and transparent reporting

### Models used
- **TF IDF + Logistic Regression** (baseline classical NLP)
- **DistilBERT** (reduced Transformer using knowledge distillation)
- **RoBERTa** (large Transformer with optimized pretraining)

### Dataset
I use a 10,000 sample subset of the Yelp Polarity dataset. One to two star reviews are labeled as churn risk, and four to five star reviews are labeled as no churn. Three star reviews are excluded.

The data is split into train, validation, and test sets using stratified sampling.

---

## 3. Implementation

All models are trained and evaluated end to end in the notebook `Churn_Prediction.ipynb`.  
The Transformer models are fine tuned using HuggingFace Transformers on a T4 GPU with a max sequence length of 128 and batch size of 32.

I saved results, charts, and metrics into an `outputs/` folder so they can be reused for analysis without re running training.

---

## 4. Model and Data Cards

### Data Card
- **Source:** Yelp Polarity (HuggingFace)
- **License:** CC BY 4.0  
- **Language:** English  
- **Limitations:** No true churn labels, domain specific, mostly standard English, no timestamps  

### Model Card (RoBERTa)
- **Type:** Transformer encoder  
- **Parameters:** ~125M  
- **Training:** 3 epochs, learning rate 2e-5  
- **Intended use:** Ranking customers by likelihood of churn from review text  
- **Not intended for:** Automated decision making, demographic inference, or financial decisions  
- **Ethical risks:** Dialect bias, incomplete context, misinterpretation of tone or sarcasm  

---

## 5. Evaluation and Results

The table below summarizes performance across accuracy, precision, recall, F1, and AUC. RoBERTa delivers the strongest results.

### Model Performance Comparison

![Model Performance Comparison](outputs/model_performance_comparison.png)

### ROC Curves

![ROC Curves](outputs/roc_curves.png)

### Confusion Matrices

![Confusion Matrices](outputs/confusion_matrices.png)

RoBERTa achieves:
- AUC: **0.987**
- Recall: **0.946**
- Precision: **0.931**
- Error rate: **6.1 percent**

The baseline performs surprisingly well due to the strong relationship between sentiment and churn, but Transformers outperform it across all metrics.

### Calibration Curves

![Calibration Curves](outputs/calibration_curves.png)

The models are reasonably well calibrated, meaning their predicted probabilities correspond closely with actual frequencies.

### Top Ten Percent Precision

This metric reflects the accuracy of the highest risk predictions. This is the group companies would intervene on first.

![Top 10 Precision](outputs/top10_precision.png)

All models exceed the required 70 percent precision. DistilBERT and RoBERTa reach **100 percent**, meaning all customers ranked in the top risk tier were true churn cases.

---

## 6. Interpretability

### Key Churn Phrases Learned by the Baseline Model

![Key Churn Phrases](outputs/key_churn_phrases.png)

Negative superlatives and strong negations predict churn, while positive superlatives predict loyalty. This aligns with linguistic theory and confirms that the models are learning meaningful features.

### Transformer Attention

Attention visualizations (in `outputs/attention/`) show that the models correctly focus on negations, intensifiers, and sentiment phrases. This supports the idea that Transformers capture structure that TF IDF cannot.

---

## 7. Bias and Ethical Considerations

I evaluated performance by review length to check for systematic bias:

- Short: 93 percent accuracy  
- Medium: 95.6 percent  
- Long: 87.5 percent  

The difference is below the 10 percent threshold we discussed in class, so there is no major fairness concern. However, the models may still have limitations based on dialect, phrasing, or writing style.

The model should never be used for automated decisions. It is best used only as a ranking signal for human review.

---

## 8. Critical Analysis

This project shows that Transformer models outperform classical NLP when identifying nuanced churn signals in text. RoBERTa delivers the strongest performance because it captures deep contextual patterns, handles negation well, and recognizes multi word expressions that TF IDF cannot.

At the same time, the strong baseline performance shows that sentiment is a major driver of churn and that simple models can still provide value for companies with limited compute resources.

The main limitations are that the labels are proxies for real churn, and the dataset comes from a single domain. Future work would use real business churn labels, additional metadata, and multilingual reviews.

---

## 9. How to Run the Project

1. Clone the repository  
2. Install the requirements using `pip install -r requirements.txt`  
3. Open `Churn_Prediction.ipynb` in Colab or VS Code  
4. Run all cells to train or evaluate the models  

All graphics are stored in `outputs/` and regenerate automatically when evaluation cells are run.

---

## 10. Repository Structure

customer-churn-nlp/
│
├── Churn_Prediction.ipynb
├── outputs/
│ ├── model_performance_comparison.png
│ ├── roc_curves.png
│ ├── confusion_matrices.png
│ ├── calibration_curves.png
│ ├── top10_precision.png
│ ├── key_churn_phrases.png
│ └── attention/
│
├── churn_model/
├── requirements.txt
└── README.md

yaml
Copy code

---

## 11. References

- Vaswani et al. "Attention Is All You Need"  
- Liu et al. "RoBERTa: A Robustly Optimized BERT Pretraining Approach"  
- Sanh et al. "DistilBERT"  
- HuggingFace Transformers documentation  
- scikit learn documentation  

---

