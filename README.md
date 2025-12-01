# Predicting Customer Churn from Reviews Using Transformer Models
---

## Executive Summary

Organizations across industries face a critical challenge in identifying customers at risk of churn before they leave. Traditional keyword based approaches and rule based systems fail to capture the subtle linguistic cues that indicate dissatisfaction. This project addresses this gap by developing and evaluating three machine learning models for predicting customer churn directly from review text, achieving up to 98.7% area under the ROC curve.

The analysis compares a classical baseline (TF-IDF with logistic regression) against two state of the art transformer architectures (DistilBERT and RoBERTa), demonstrating that modern neural language models can extract nuanced semantic patterns that traditional methods miss. Beyond predictive accuracy, this work emphasizes model interpretability through attention visualization, calibration analysis, and linguistic pattern extraction, ensuring that predictions are both accurate and explainable for business stakeholders.

<img width="2685" height="1481" alt="model_performance_comparison" src="https://github.com/user-attachments/assets/7f8c0173-ec15-40b8-9037-7834f43bf151" />


---

## Problem Statement and Business Context

Customer churn represents a significant revenue risk for service oriented businesses. Research indicates that acquiring new customers costs five to seven times more than retaining existing ones. While organizations collect thousands of customer reviews daily, manually analyzing this volume of unstructured text to identify at risk customers is impractical.

Existing approaches rely on simple sentiment scoring or keyword matching, which prove inadequate for several reasons. First, sentiment polarity alone does not reliably predict churn behavior; a customer may leave positive feedback while still planning to switch providers. Second, keyword based systems miss context dependent meaning, such as negation ("not bad" versus "bad") or sarcasm. Third, these methods cannot capture the subtle differences between phrases like "considering alternatives" and "looking at alternatives," which carry distinct implications for churn risk.

### Research Objectives

This project develops a binary classification system that predicts customer churn risk from review text with the following technical requirements:

1. **Predictive Performance:** Achieve area under the ROC curve of at least 0.85, substantially exceeding random classification
2. **Precision Targeting:** Maintain at least 70% precision on the top 10% of high risk predictions to enable cost effective intervention
3. **Operational Feasibility:** Ensure inference latency below 100 milliseconds for real time scoring applications

### Contribution to Domain Knowledge

This work contributes to both practical applications and theoretical understanding. Practically, it demonstrates how transformer based natural language processing can be deployed for customer analytics at scale. Theoretically, it provides empirical evidence of the performance gains achievable through contextual embeddings compared to traditional bag of words representations, particularly for tasks requiring nuanced understanding of sentiment and intent.

---

## Theoretical Foundation and Course Integration

This project synthesizes multiple concepts from advanced machine learning and natural language processing theory, directly applying course material to a real world business problem.

### Transformer Architecture and Self Attention

The transformer architecture, introduced by Vaswani et al. (2017), revolutionized natural language processing by replacing recurrent connections with self attention mechanisms. Unlike sequential models that process text left to right, self attention allows each token to directly attend to every other token in the sequence, capturing long range dependencies more effectively.

The self attention mechanism computes three learned projections for each input token: queries, keys, and values. Attention weights are calculated as the softmax of scaled dot products between queries and keys, then used to weight the value vectors. This allows the model to dynamically determine which parts of the input are most relevant for each token's representation.

Both DistilBERT and RoBERTa build on this foundation but with different design choices. DistilBERT (Sanh et al., 2019) uses knowledge distillation to create a smaller, faster model that retains 97% of BERT's performance with 40% fewer parameters. RoBERTa (Liu et al., 2019) improves on BERT through optimized pretraining, including longer training, larger batches, and removal of the next sentence prediction objective.

### Transfer Learning and Fine Tuning

Rather than training from scratch, this project employs transfer learning, a paradigm where models pretrained on large corpora are adapted to specific downstream tasks. The pretrained models have already learned general linguistic representations from billions of tokens, which we then specialize for churn classification through fine tuning.

Fine tuning updates all model parameters on the labeled churn dataset while starting from the pretrained weights. This approach requires far less labeled data than training from scratch and typically yields superior performance, as the model leverages both general language understanding and task specific patterns.

### Evaluation Methodology and Metrics

Model evaluation extends beyond simple accuracy to capture multiple dimensions of performance relevant to the business use case.

**Area Under the ROC Curve (AUC)** measures the model's ability to rank positive examples higher than negative examples across all possible decision thresholds. An AUC of 0.5 indicates random performance, while 1.0 represents perfect discrimination. This metric proves particularly valuable for imbalanced datasets and threshold independent evaluation.

**Precision and Recall** trade offs matter significantly in practice. High precision minimizes false positives (customers incorrectly flagged as churn risks), reducing wasted intervention costs. High recall minimizes false negatives (customers who churn without warning), capturing more revenue at risk. The F1 score provides a harmonic mean of these metrics.

**Calibration Curves** assess whether predicted probabilities align with empirical frequencies. A well calibrated model's prediction of 70% churn risk should correspond to roughly 70% of such cases actually churning. Calibration matters when predictions inform decision thresholds or cost benefit analysis.

**Top K Precision** focuses evaluation on the highest risk predictions, which typically receive the most intensive (and expensive) intervention in practice. Achieving 100% precision on the top 10% of predictions means every customer flagged in this tier actually represents a true churn risk.

### Bias Detection and Fairness Considerations

Machine learning models can inadvertently encode and amplify biases present in training data. This analysis includes systematic bias detection along several dimensions:

**Length Bias:** Models might perform differently on short versus long reviews due to information density or writing style correlations. Analysis across review length categories (under 20 words, 20 to 100 words, over 100 words) reveals any systematic performance disparities.

**Class Balance:** Severely imbalanced datasets can lead models to achieve high accuracy simply by predicting the majority class. Stratified sampling and balanced performance metrics mitigate this risk.

**Linguistic Bias:** Models trained primarily on standard English may underperform on non-standard dialects, slang, or code switching. While the Yelp dataset limits assessment of this dimension, documentation of this limitation informs appropriate use cases.

---

## Dataset and Experimental Design

### Data Source and Characteristics

The analysis uses the Yelp Polarity dataset, a widely used benchmark for sentiment classification tasks in natural language processing research. This dataset originates from the Yelp Dataset Challenge and contains user reviews with star ratings ranging from 1 to 5.

**Dataset Specifications:**
- **Source:** Yelp Dataset Challenge via HuggingFace Datasets
- **Sample Size:** 10,000 reviews (stratified random sample)
- **License:** Creative Commons Attribution 4.0
- **Language:** English
- **Domain:** Service industry reviews (restaurants, businesses)

**[INSERT GRAPHIC: Sample reviews or class distribution visualization]**

### Label Construction and Task Formulation

The original Yelp Polarity dataset provides binary sentiment labels (positive or negative). This project remaps these labels to churn risk categories based on the well established relationship between customer satisfaction and retention:

- **1 to 2 Star Reviews → Churn Risk (Label 1):** Reviews with ratings of 1 or 2 stars indicate substantial dissatisfaction, correlating with high probability of customer attrition
- **4 to 5 Star Reviews → No Churn (Label 0):** Reviews with ratings of 4 or 5 stars indicate satisfaction, correlating with continued patronage

This mapping transforms sentiment classification into churn prediction while maintaining the same underlying classification task. The exclusion of 3 star reviews (neutral sentiment) creates a cleaner binary decision boundary, though it limits generalization to ambiguous cases.

### Data Partitioning Strategy

Proper data partitioning prevents overfitting and enables unbiased performance estimation. The 10,000 review sample undergoes stratified random splitting:

- **Training Set:** 8,000 samples (80%)
- **Validation Set:** 2,000 samples (10%)  
- **Test Set:** 2,000 samples (10%)

Stratification ensures that class proportions (approximately 50% churn rate) remain consistent across all splits. The validation set guides model selection and hyperparameter tuning without touching the test set, which provides final performance estimates on completely unseen data.

All splits use a fixed random seed (42) for reproducibility, ensuring that results can be independently verified.

### Data Quality and Limitations

**Strengths:**
- Large sample size provides statistical power
- Balanced class distribution (49.8% churn rate) prevents trivial majority class predictions  
- Publicly available dataset enables reproducibility
- Human generated reviews reflect authentic language patterns

**Limitations and Biases:**
- **Selection Bias:** Yelp reviewers may differ systematically from non-reviewers in sentiment or writing style
- **Temporal Dynamics:** The dataset lacks timestamps, preventing analysis of how churn language evolves over time
- **Missing Context:** Reviews appear without customer history, demographics, or service details that might inform churn prediction in practice
- **Domain Specificity:** Patterns learned from restaurant and service reviews may not transfer to other industries
- **Sentiment Proxy:** Star ratings approximate but do not directly measure actual churn behavior; customers may leave positive reviews while switching providers or vice versa
- **Language Variety:** Predominantly standard English reviews may not represent diverse linguistic communities

These limitations inform appropriate use cases and suggest directions for future work.

---

## Methodology and Model Architecture

### Baseline Model: TF-IDF with Logistic Regression

The baseline employs Term Frequency Inverse Document Frequency (TF-IDF) vectorization coupled with logistic regression, representing best practices in classical text classification.

**TF-IDF Vectorization:**
- **Feature Extraction:** Converts text into numerical vectors by weighting term frequency within documents against inverse document frequency across the corpus
- **Vocabulary Size:** 5,000 most frequent terms
- **N-gram Range:** Unigrams and bigrams (1 to 2 consecutive tokens)
- **Document Frequency Bounds:** Minimum document frequency of 5, maximum of 80% to filter very rare and very common terms

**Logistic Regression Configuration:**
- **Algorithm:** L2 regularized logistic regression with balanced class weights
- **Regularization:** Inverse regularization strength determined through validation
- **Optimization:** Limited memory BFGS with maximum 1,000 iterations
- **Class Weighting:** Automatic balancing inversely proportional to class frequencies

This baseline serves multiple purposes. First, it establishes a competitive benchmark rooted in established methods. Second, it provides a point of comparison for assessing whether the added complexity of transformer models yields meaningful performance gains. Third, it enables computational cost comparison (training time, inference latency).

### Transformer Model: DistilBERT

DistilBERT represents an efficient variant of BERT designed for production deployment scenarios requiring lower latency and memory footprint.

**Architecture Specifications:**
- **Base Model:** distilbert-base-uncased (pretrained on English Wikipedia and BookCorpus)
- **Encoder Layers:** 6 transformer blocks (versus 12 in BERT base)
- **Hidden Dimension:** 768
- **Attention Heads:** 12 per layer
- **Parameters:** Approximately 66 million
- **Tokenization:** WordPiece with 30,000 token vocabulary

**Fine Tuning Configuration:**
- **Sequence Length:** 128 tokens maximum (truncation applied)
- **Batch Size:** 32 samples per gradient update
- **Training Epochs:** 3 complete passes through training data
- **Learning Rate:** 2×10⁻⁵ with linear warmup (500 steps)
- **Optimizer:** AdamW with weight decay 0.01
- **Mixed Precision:** FP16 training enabled for GPU efficiency
- **Early Stopping:** Validation F1 score monitoring with patience of 1 epoch

**Training Infrastructure:**
- **Hardware:** Tesla T4 GPU (15.83 GB memory) via Google Colab
- **Training Time:** Approximately 2.4 minutes
- **Framework:** HuggingFace Transformers 4.x with PyTorch 2.9.0

### Transformer Model: RoBERTa

RoBERTa (Robustly Optimized BERT Pretraining Approach) improves upon BERT through several architectural and training modifications.

**Architecture Specifications:**
- **Base Model:** roberta-base (pretrained on 160GB of text including Common Crawl)
- **Encoder Layers:** 12 transformer blocks
- **Hidden Dimension:** 768
- **Attention Heads:** 12 per layer  
- **Parameters:** Approximately 125 million
- **Tokenization:** Byte-level Byte Pair Encoding with 50,000 token vocabulary

**Fine Tuning Configuration:**
Identical hyperparameters to DistilBERT for controlled comparison:
- **Sequence Length:** 128 tokens maximum
- **Batch Size:** 32 samples per gradient update
- **Training Epochs:** 3
- **Learning Rate:** 2×10⁻⁵ with linear warmup (500 steps)
- **Optimizer:** AdamW with weight decay 0.01

**Training Infrastructure:**
- **Hardware:** Tesla T4 GPU via Google Colab
- **Training Time:** Approximately 4.8 minutes
- **Framework:** HuggingFace Transformers 4.x with PyTorch 2.9.0

The increased parameter count and more extensive pretraining corpus position RoBERTa to potentially capture more nuanced linguistic patterns than DistilBERT, albeit at the cost of longer training and inference times.

### Experimental Controls and Reproducibility

All experiments employ strict controls to ensure fair comparison and reproducibility:

- **Random Seed:** Fixed at 42 across NumPy, PyTorch, and data splitting operations
- **Data Splits:** Identical train/validation/test partitions across all models
- **Evaluation Protocol:** Identical metrics, thresholds, and reporting for all models
- **Hardware:** All models trained and evaluated on the same GPU configuration
- **Software Versions:** Locked dependency versions (PyTorch 2.9.0, Transformers 4.x, scikit-learn 1.x)

These controls isolate model architecture as the primary variable, enabling valid performance comparisons.

---

## Results and Performance Analysis

### Comprehensive Model Comparison

**[INSERT GRAPHIC: model_performance_comparison.png - this should be prominently displayed]**

The following table presents test set performance across all models and metrics:

| **Metric** | **TF-IDF + Logistic Regression** | **DistilBERT** | **RoBERTa** | **Best Improvement** |
|------------|----------------------------------|----------------|-------------|---------------------|
| Accuracy | 90.7% | 91.6% | **93.9%** | +3.2 percentage points |
| Precision | 91.1% | 91.7% | **93.1%** | +2.0 percentage points |
| Recall | 90.0% | 91.2% | **94.6%** | +4.6 percentage points |
| F1 Score | 90.5% | 91.4% | **93.9%** | +3.4 percentage points |
| **AUC** | 96.9% | 97.2% | **98.7%** | +1.8 percentage points |
| Training Time | 7.2 seconds | 145 seconds | 290 seconds | - |
| Inference Latency | 0.58 ms | 5.6 ms | 13.5 ms | - |

**Key Findings:**

1. **RoBERTa achieves superior performance** across all predictive metrics, establishing it as the best model for maximizing churn detection accuracy

2. **Transformer models demonstrate consistent improvements** over the classical baseline, with gains ranging from 1.8% (AUC) to 4.6% (recall)

3. **Recall improvements prove most substantial** (4.6 percentage points), indicating that transformers excel at identifying true churn cases that the baseline misses

4. **Accuracy improvements appear modest** (3.2 percentage points), but this reflects the already strong baseline (90.7%) rather than weak transformer performance

5. **The baseline achieves surprisingly high performance** (96.9% AUC), suggesting that review sentiment correlates strongly with churn risk and that even simple methods capture much of this signal

6. **Computational costs scale with model complexity**, with RoBERTa requiring 40× longer training than the baseline but still maintaining sub-100ms inference latency

**[INSERT GRAPHIC: roc_curves.png - ROC curves for all three models]**

### ROC Curve Analysis

The receiver operating characteristic curves visualize each model's discrimination ability across all possible decision thresholds. RoBERTa's curve dominates the others, tracking closer to the top left corner (perfect classification) throughout its range. The area under these curves quantifies overall ranking quality independent of threshold selection.

The baseline's strong ROC curve (96.9% AUC) demonstrates that TF-IDF captures meaningful signal, but the transformers' improvements show that contextual understanding adds value. The gap between DistilBERT and RoBERTa suggests that additional model capacity and pretraining data continue to improve performance even at this scale.

**[INSERT GRAPHIC: confusion_matrices.png - confusion matrices for all three models]**

### Error Analysis Through Confusion Matrices

Confusion matrices reveal the distribution of prediction errors across true positive, true negative, false positive, and false negative categories.

**RoBERTa Confusion Matrix (Best Model):**
- **True Negatives:** 942 (correctly identified no churn cases)
- **False Positives:** 69 (incorrectly flagged as churn)
- **False Negatives:** 53 (missed churn cases)  
- **True Positives:** 936 (correctly identified churn cases)

**Error Rate:** 6.1% (122 errors out of 2,000 test cases)

**Error Pattern Insights:**

The relatively balanced distribution of false positives and false negatives (69 versus 53) indicates that the model does not systematically bias toward either over-prediction or under-prediction of churn. False positives incur costs of unnecessary intervention, while false negatives represent missed revenue protection opportunities. The near balance suggests the model's default threshold (0.5 probability) provides reasonable trade-offs.

Qualitative examination of misclassified examples reveals several error patterns:

1. **Ambiguous Sentiment:** Reviews containing both positive and negative elements (e.g., "great food but terrible service") confuse the model
2. **Sarcasm Detection Failures:** Sarcastic positive language in otherwise negative reviews occasionally fools the classifier
3. **Context-Dependent Phrases:** Expressions like "nothing special" may appear in both churn and no-churn contexts
4. **Extreme Brevity:** Very short reviews (under 10 words) provide limited signal for any method

### Top Decile Precision Analysis

**[INSERT GRAPHIC: top10_precision.png - bar chart showing top 10% precision across models]**

Business applications often prioritize the highest risk predictions for intensive intervention. The top 10% precision metric evaluates model performance on this critical subset.

**Results:**
- **Baseline:** 99.0% precision (198 out of 200 truly churned)
- **DistilBERT:** 100% precision (200 out of 200 truly churned)
- **RoBERTa:** 100% precision (200 out of 200 truly churned)

**Implications:**

Perfect precision on the top decile demonstrates that both transformer models achieve the target threshold (70%) with substantial margin. In practice, this means that prioritizing intervention on the 200 customers with highest predicted churn probability would successfully target actual at-risk customers with no wasted effort.

This performance level proves particularly valuable for high-cost interventions (personalized retention offers, account manager outreach) where false positives significantly impact return on investment. The baseline's near-perfect performance (99%) indicates that simple methods already provide strong signal in this regime, though transformers eliminate the remaining errors.

### Calibration Analysis

<img width="1482" height="1180" alt="calibration_curves (1)" src="https://github.com/user-attachments/assets/97cc280e-5f3c-48c5-90f7-355a7b754ecc" />

Calibration curves plot predicted probabilities against empirical frequencies, revealing whether the model's confidence estimates align with reality. A perfectly calibrated model's predictions of X% probability should correspond to roughly X% of such cases being positive.

**Calibration Findings:**

All three models demonstrate reasonable calibration, with predicted probabilities tracking empirical frequencies relatively closely. The transformer models show slightly better calibration than the baseline, particularly in the mid-probability range (0.3 to 0.7) where uncertain predictions concentrate.

Well-calibrated probabilities enable threshold optimization for specific cost-benefit scenarios. For example, if intervention costs $10 and retaining a customer generates $100 profit, the optimal threshold becomes any probability above 10% (where expected value turns positive). Poorly calibrated models would require probability rescaling before such decision analysis.

### Inference Latency Comparison

<img width="1482" height="882" alt="inference_latency_comparison" src="https://github.com/user-attachments/assets/d118e4a1-258a-4720-84a1-33f605ebe763" />

Real-time applications require predictions within tight latency budgets. The following latencies represent average inference time per single review:

- **Baseline:** 0.58 milliseconds
- **DistilBERT:** 5.6 milliseconds  
- **RoBERTa:** 13.5 milliseconds

**Performance-Latency Trade-offs:**

All models comfortably meet the 100 millisecond target, supporting real-time scoring applications. The baseline's sub-millisecond latency enables throughput exceeding 1,000 predictions per second on a single core, suitable for batch processing scenarios.

DistilBERT's 10× higher latency versus the baseline represents the cost of contextual understanding, but its 2.4× speed advantage over RoBERTa demonstrates the value of model compression techniques. For applications requiring both high accuracy and low latency, DistilBERT offers an attractive middle ground.

These measurements assume GPU inference (Tesla T4). CPU inference would increase latencies approximately 5 to 10 fold, potentially necessitating DistilBERT over RoBERTa for high-throughput CPU deployments.

---

## Model Interpretability and Linguistic Analysis

### Feature Importance from TF-IDF Coefficients

<img width="2385" height="1180" alt="key_churn_phrases" src="https://github.com/user-attachments/assets/9db9cf74-6a5a-426c-be11-9423a7cb761a" />

The logistic regression baseline's learned coefficients reveal which words most strongly predict each class. Higher magnitude coefficients indicate stronger predictive power.

**Top Churn Indicators (Negative Reviews):**
1. "not" (coefficient: -5.76)
2. "no" (coefficient: -4.36)
3. "worst" (coefficient: -3.52)
4. "bad" (coefficient: -3.48)
5. "bland" (coefficient: -3.22)
6. "terrible" (coefficient: -3.17)
7. "rude" (coefficient: -3.06)
8. "horrible" (coefficient: -2.95)
9. "mediocre" (coefficient: -2.80)
10. "overpriced" (coefficient: -2.54)

**Top No-Churn Indicators (Positive Reviews):**
1. "great" (coefficient: +6.13)
2. "delicious" (coefficient: +5.54)
3. "amazing" (coefficient: +4.17)
4. "excellent" (coefficient: +3.92)
5. "love" (coefficient: +3.91)
6. "best" (coefficient: +3.79)
7. "awesome" (coefficient: +3.79)
8. "perfect" (coefficient: +2.81)
9. "friendly" (coefficient: +2.81)
10. "favorite" (coefficient: +2.49)

**Linguistic Insights:**

The strongest churn predictors include absolute negations ("not", "no") and extreme negative descriptors ("worst", "terrible", "horrible"). This aligns with linguistic theory on sentiment intensification, where speakers escalate language to emphasize dissatisfaction.

Conversely, no-churn predictions rely heavily on superlatives ("best", "perfect") and emotional language ("love", "favorite", "amazing"). The prominence of "delicious" reflects the restaurant-heavy composition of Yelp reviews.

Bigrams provide additional context. Phrases like "not worth", "never again", and "waste of" strongly predict churn, while "highly recommend", "come back", and "can't wait" predict loyalty. These multi-word expressions capture sentiment more precisely than individual words.

### Attention Visualization

**[INSERT GRAPHIC: Example attention heatmap from outputs/attention/ folder]**

Transformer attention weights reveal which input tokens the model focuses on when making predictions. Attention visualization generates heatmaps where darker colors indicate higher attention weights between token pairs.

**Attention Pattern Observations:**

1. **Negation Handling:** The model attends strongly from sentiment words to preceding negations (e.g., "not" → "good"), suggesting successful capture of negation scope

2. **Amplifier Detection:** Intensifiers like "very", "extremely", and "absolutely" receive high attention from the sentiment words they modify

3. **Multi-Hop Reasoning:** In complex sentences, attention patterns show indirect connections spanning multiple tokens, indicating compositional understanding

4. **Punctuation Sensitivity:** Attention weights change noticeably around sentence boundaries, suggesting the model uses structural cues

These patterns validate that the transformer leverages contextual relationships rather than treating text as an unordered bag of words. However, attention weights do not perfectly correspond to causal importance; high attention to a token does not necessarily mean that token drives the prediction.

### Bias Analysis Across Review Lengths

A systematic bias analysis examines whether model performance varies based on review length, which could indicate unfair treatment of different writing styles.

**Performance by Review Length:**

| **Category** | **Sample Count** | **Accuracy** |
|--------------|------------------|--------------|
| Short (under 20 words) | 115 reviews | 93.0% |
| Medium (20 to 100 words) | 929 reviews | 95.6% |
| Long (over 100 words) | 956 reviews | 87.5% |

**Bias Assessment:**

The maximum performance difference across length categories measures 8.1 percentage points (between medium and long reviews). Using a conservative threshold of 10 percentage points to flag substantial bias, this analysis finds no evidence of systematic length discrimination.

The slightly lower performance on long reviews may reflect greater complexity and nuance rather than bias. Longer reviews often contain mixed sentiment, qualifying statements, and contextual detail that complicates classification for any method.

The model's strong performance on short reviews (93.0%) demonstrates robustness to limited context, though this category contains too few samples (115) for high-confidence conclusions.

---

## Critical Analysis and Discussion

### Project Impact and Business Value

This work demonstrates the feasibility of deploying transformer based natural language processing for customer churn prediction at scale. The achieved performance levels (98.7% AUC, 100% top decile precision) exceed requirements for practical deployment, suggesting that organizations could realize measurable value from implementation.

**Quantitative Impact Estimation:**

Consider a business with 100,000 customers, 10% annual churn rate, and $500 lifetime value per retained customer. A perfect churn prediction system would identify all 10,000 at risk customers, enabling targeted retention efforts. At 30% intervention success rate and $50 per intervention cost, the value calculation becomes:

- Customers saved: 10,000 × 30% = 3,000
- Revenue protected: 3,000 × $500 = $1.5 million
- Intervention cost: 10,000 × $50 = $500,000  
- Net value: $1 million annually

The RoBERTa model's 94.6% recall would identify 9,460 of the 10,000 churners, capturing 94.6% of this potential value ($946,000) while maintaining cost discipline through high precision.

### What the Analysis Reveals

**Transformer Advantages:**

The consistent performance improvements across all metrics validate theoretical expectations about contextual embeddings. Transformers excel specifically in areas where bag of words representations struggle:

1. **Negation Handling:** Understanding that "not bad" differs fundamentally from "bad"
2. **Long Range Dependencies:** Connecting sentiment words to their targets across sentence spans
3. **Compositional Semantics:** Combining word meanings based on syntactic structure
4. **Ambiguity Resolution:** Using context to disambiguate polysemous words

**Baseline Strength:**

The TF-IDF baseline's surprisingly strong performance (96.9% AUC) reveals that review sentiment correlates powerfully with churn risk, and that simple frequency based features capture much of this signal. This finding has practical implications: organizations with limited machine learning infrastructure can still achieve substantial value from classical methods, reserving transformer deployment for scenarios requiring marginal performance gains.

**Model Selection Considerations:**

The choice between baseline, DistilBERT, and RoBERTa depends on deployment constraints:

- **Baseline:** Optimal for batch processing, limited ML infrastructure, or scenarios where 96.9% AUC suffices
- **DistilBERT:** Best balance of performance (97.2% AUC) and efficiency (5.6ms latency) for real-time applications  
- **RoBERTa:** Maximum performance (98.7% AUC) for high-value customer segments where accuracy justifies computational cost

### Limitations and Threats to Validity

**Dataset Limitations:**

1. **Sentiment-Churn Gap:** Star ratings approximate but do not directly measure churn behavior. Customers may leave positive reviews while switching providers (e.g., due to relocation) or negative reviews while remaining loyal (due to isolated incidents)

2. **Selection Bias:** Yelp reviewers represent a non-random sample of customers. Research shows that extremely satisfied and extremely dissatisfied customers disproportionately write reviews, creating a bimodal distribution not representative of the full customer base

3. **Temporal Staleness:** The dataset lacks timestamps, preventing assessment of whether churn language evolves over time or whether model performance degrades as language trends shift

4. **Domain Specificity:** Patterns learned from restaurant and service reviews may not transfer to other industries (retail, software, healthcare) with different customer touchpoints and churn drivers

**Model Limitations:**

1. **Missing Context:** Real churn prediction systems would incorporate customer history (purchase frequency, tenure, service usage), demographics, and behavioral signals. Text alone provides incomplete information

2. **Language Restriction:** English-only training limits applicability to global markets and multilingual customer bases

3. **Interpretability Challenges:** While attention weights provide some insight, they do not constitute complete explanations of model reasoning. Recent research shows attention does not reliably indicate causal importance

4. **Adversarial Vulnerability:** Neither baseline nor transformer models include explicit robustness mechanisms against intentionally deceptive text

**Experimental Limitations:**

1. **Sample Size:** The 10,000 review sample, while adequate for initial validation, represents only 1.8% of the available Yelp Polarity data. Scaling to the full dataset might reveal different performance characteristics

2. **Hyperparameter Search:** Limited computational budget prevented exhaustive hyperparameter optimization. More extensive tuning might narrow the performance gap between models or reveal optimal configurations outside tested ranges

3. **Single Dataset:** Evaluation on one dataset limits generalizability claims. Cross-domain validation would strengthen conclusions

### Ethical Considerations and Responsible Use

**Bias and Fairness:**

While length bias analysis shows no systematic discrimination, other bias dimensions remain unexamined. Potential concerns include:

- **Dialect Bias:** Models trained on standard English may underperform on African American Vernacular English, regional dialects, or non-native speaker English
- **Vocabulary Bias:** Technical jargon, industry specific terminology, or age-cohort language might affect predictions independently of actual churn risk  
- **Demographic Proxies:** Review language patterns may correlate with demographic attributes (age, education, region), creating indirect discrimination even without explicit demographic features

**Appropriate Use Cases:**

This model should augment rather than replace human judgment in customer retention decisions. Appropriate applications include:

- **Prioritization:** Ranking customer support queue by predicted churn risk
- **Triggering Alerts:** Flagging high-risk customers for account manager review
- **Trend Analysis:** Identifying emerging dissatisfaction themes across the customer base

**Inappropriate Applications:**

- **Automated Penalties:** Using predictions to restrict service, raise prices, or terminate accounts without human review
- **Sole Decision Making:** Making high-stakes retention decisions based purely on model output
- **Protected Decisions:** Any application where prediction errors could affect protected classes differently (employment, housing, credit)

**Transparency Requirements:**

Organizations deploying this system should:
- Disclose to customers that their feedback may inform retention efforts
- Provide mechanisms for customers to correct misclassifications
- Regularly audit predictions for fairness across demographic groups
- Document model limitations and confidence bounds for stakeholders

### Future Research Directions

**Methodological Extensions:**

1. **Ensemble Methods:** Combining TF-IDF and transformer predictions through stacking or voting might capture complementary strengths
2. **Active Learning:** Selecting the most informative examples for labeling could improve sample efficiency
3. **Multi-Task Learning:** Joint training on sentiment analysis, aspect extraction, and churn prediction might improve representation quality
4. **Temporal Modeling:** Incorporating review timing and sequence patterns could capture churn dynamics

**Dataset Enhancements:**

1. **Verified Churn Labels:** Partnering with businesses to obtain ground-truth churn outcomes would eliminate the sentiment proxy limitation
2. **Multi-Domain Evaluation:** Testing on reviews from diverse industries (software, retail, healthcare) would assess generalization
3. **Longitudinal Data:** Time-stamped reviews would enable study of language evolution and concept drift
4. **Multilingual Extension:** Training on non-English reviews would expand applicability

**Deployment Considerations:**

1. **Model Compression:** Quantization, pruning, or distillation could reduce RoBERTa's inference latency for production deployment
2. **Online Learning:** Continuous model updating with new reviews would maintain performance as language evolves
3. **Explainable AI:** SHAP values, influence functions, or counterfactual explanations could provide stakeholder-friendly interpretations beyond attention weights
4. **A/B Testing:** Controlled experiments comparing retention rates with and without model-guided interventions would quantify business impact

---

## Technical Implementation and Reproducibility

### Software Environment

**Core Dependencies:**
- Python 3.10
- PyTorch 2.9.0 with CUDA 12.6 support
- HuggingFace Transformers 4.x
- scikit-learn 1.x  
- pandas, numpy, matplotlib, seaborn

**Hardware:**
- Google Colab Tesla T4 GPU (15.83 GB memory)
- CUDA enabled for mixed precision training

**Reproducibility Configuration:**
All random operations use fixed seed 42, ensuring deterministic data splits and model initialization. Complete dependency versions appear in the project requirements file.

### Code Availability and Usage

The complete analysis pipeline exists in a single Jupyter notebook (`Churn_Prediction.ipynb`) with clearly delineated sections:

1. **Configuration and Setup:** Library imports, hyperparameter definitions, GPU detection
2. **Data Loading:** Yelp Polarity dataset download and stratified splitting  
3. **Baseline Training:** TF-IDF vectorization and logistic regression fitting
4. **Transformer Fine-Tuning:** DistilBERT and RoBERTa training loops
5. **Evaluation:** Metric computation, visualization generation
6. **Interpretability:** Attention visualization, coefficient analysis, bias detection

**Execution Instructions:**

To reproduce results:
1. Upload notebook to Google Colab
2. Select Runtime → Change runtime type → GPU (T4)
3. Execute all cells sequentially
4. Total runtime approximately 15 to 20 minutes

No manual data download required; the HuggingFace datasets library handles acquisition automatically.

**Model Artifacts:**

Trained models save to the following locations:
- TF-IDF vectorizer and logistic regression: `churn_model/tfidf_vectorizer.pkl`, `churn_model/lr_model.pkl`
- DistilBERT: `churn_model/distilbert/` (excluded from repository due to 250 MB size)
- RoBERTa: `churn_model/roberta/` (excluded from repository due to 500 MB size)

The transformer model directories can be regenerated by executing the training cells, which automatically save checkpoints.

### Data Access and Licensing

**Dataset Access:**
```python
from datasets import load_dataset
dataset = load_dataset("yelp_polarity")
```

**Licensing:**
- Yelp Polarity Dataset: Creative Commons Attribution 4.0 International
- DistilBERT Model: Apache License 2.0
- RoBERTa Model: MIT License  
- scikit-learn: BSD 3-Clause License
- Project Code: MIT License

All components permit academic and commercial use with proper attribution.

---

## Conclusion

This project demonstrates that state of the art transformer models provide measurable improvements in customer churn prediction from review text compared to classical natural language processing baselines. RoBERTa achieves 98.7% area under the ROC curve and 100% precision on the top 10% of predictions, exceeding all target performance thresholds with substantial margin.

Beyond predictive accuracy, this work emphasizes responsible machine learning practices including bias analysis, interpretability investigation, and transparent documentation of limitations. The attention visualizations and coefficient analyses reveal that models focus on linguistically meaningful patterns rather than spurious correlations, building confidence in their appropriateness for decision support applications.

The results validate theoretical expectations about contextual embeddings while also revealing that classical methods remain surprisingly competitive (96.9% AUC). This suggests a nuanced deployment strategy where organizations balance performance requirements against computational constraints, reserving transformer models for high-value applications where marginal accuracy gains justify increased complexity.

Future work should address current limitations through verified churn labels, multi-domain evaluation, and deployment pilots with real-world A/B testing. These extensions would strengthen both the scientific understanding of language based churn prediction and its practical applicability across industries.

### Success Metrics Achievement

**Target Performance:**
- Area Under ROC Curve ≥ 0.85: **Achieved 0.987** (16% above target)
- Top 10% Precision ≥ 70%: **Achieved 100%** (43% above target)  
- Inference Latency < 100ms: **Achieved 5.6ms** (94% under threshold)

**All objectives met or substantially exceeded.**

---

## References

Complete citations and resource links appear in [REFERENCES.md](REFERENCES.md), including:

- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*
- Devlin, J., et al. (2018). BERT: Pre-training of deep bidirectional transformers. *NAACL*  
- Sanh, V., et al. (2019). DistilBERT: A distilled version of BERT. *NeurIPS Workshop*
- Liu, Y., et al. (2019). RoBERTa: A robustly optimized BERT pretraining approach. *arXiv*
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS*

---

## Author and Contact

**Ashley Stevens**  
Master of Science in Applied Data Science  
University of Southern California  
GitHub: [@steveam1](https://github.com/steveam1)  
Repository: [customer-churn-nlp](https://github.com/steveam1/customer-churn-nlp)

---

*Document prepared for DSCI 552 final project presentation  
December 2024*

