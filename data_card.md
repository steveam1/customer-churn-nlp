# Data Card for Customer Churn Prediction Project

This data card documents the dataset used for developing the customer churn prediction models. Its purpose is to promote transparency about the data source, labeling process, preprocessing, and limitations.

---

## Dataset Summary

The project uses the Yelp Polarity dataset, which consists of user written restaurant reviews labeled as positive or negative. For this project:

- positive reviews were mapped to no churn  
- negative reviews were mapped to churn  

This mapping creates a binary churn classification dataset used for training and evaluation.

---

## Data Source

The dataset is publicly available through HuggingFace Datasets. It was originally created for sentiment analysis research and contains no personal identifying information.

---

## Data Size

The subset used in this project includes:

- 10,000 samples for training  
- 2,000 samples for testing  
- balanced classes  

Each sample contains:

- a raw text review  
- an associated sentiment label  

---

## Data Fields

- **text**: the written review  
- **label**: binary value representing mapped churn status  

---

## Collection Method

The data was collected from publicly available Yelp reviews. Yelp users post reviews voluntarily. The dataset creators stripped identifying information and released the dataset for academic and research use.

---

## Data Preprocessing

Minimal preprocessing was applied:

- no removal of punctuation  
- no lowercasing  
- no stop word filtering  

This was intentional so that transformer tokenizers could process the text as raw language and learn contextual embeddings.

TF IDF preprocessing was performed only for the classical baseline model, which uses one and two gram features.

---

## Labeling Process

Sentiment labels were mapped to churn labels:

- negative sentiment equals churn  
- positive sentiment equals no churn  

This means the labels represent opinion based dissatisfaction rather than true customer behavior.

---

## Potential Data Quality Issues

- reviews may contain sarcasm  
- reviews sometimes contain mixed sentiment within a single text  
- reviews may contain domain specific vocabulary that does not transfer to other industries  
- sentiment is only a proxy for churn, not actual customer abandonment  

These issues should be considered when interpreting model predictions.

---

## Ethical Considerations

### Privacy  
The dataset does not contain personal identifying information and is cleared for research use.

### Bias  
Because the dataset lacks demographic information, fairness across demographic groups cannot be evaluated. Review length bias was tested and results were within acceptable thresholds.

### Proxy labeling  
The labels represent text sentiment, not real churn behavior. This should be kept in mind when deploying models.

---

## Recommended Uses

- research on text based churn prediction  
- comparison of classical and transformer NLP models  
- experimentation with interpretability tools  
- coursework and educational demonstrations  

---

## Not Recommended For

- real business decisions without retraining on true churn data  
- demographic fairness analysis  
- prediction of customer behavior outside the review domain  

---

## Citation

If referencing this dataset configuration, please cite:

**Data Card for Churn Prediction Project  
Version 1.0**
