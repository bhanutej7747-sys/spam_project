# Project 4: Spam Email Classification

## Objective
Build a machine learning model to classify SMS messages as **Spam** or
**Ham**.

## Dataset
**SMS Spam Collection Dataset** — 5,572 real SMS messages before
cleaning (5,169 after removing 403 duplicates), each labeled ham or
spam.

---

## Task 1: Load, Clean and Preprocess the Dataset
- Loaded the dataset (5,572 rows, 2 columns: `label`, `message`)
- Checked data types — no missing values found
- Found and removed **403 duplicate rows** → 5,169 unique messages
- Preprocessed each message: lowercased, removed URLs, punctuation,
  and digits, removed stopwords, and applied **Porter Stemming** to
  reduce words to their root form (e.g. "winning" -> "win")

## Task 2: Convert Text to Numerical Features + EDA
- **EDA findings**:
  - Class distribution is imbalanced: 4,516 ham vs 653 spam
  - Spam messages average **137.7 characters**, almost double ham
    messages at **70.9 characters** — spam tends to be longer due to
    promotional text and links
  - See `eda_class_distribution.png` and `eda_message_length.png`
- Converted cleaned text into numerical features using **TF-IDF**
  (Term Frequency-Inverse Document Frequency), capped at 3,000
  features, producing a (5169 x 3000) matrix

## Task 3: Train-Test Split
- Split the data 80/20 (stratified, so both sets keep the same
  ham/spam ratio)
- Training set: 4,135 messages | Testing set: 1,034 messages

## Task 4: Train Naive Bayes Classifier
- Trained a **Multinomial Naive Bayes** classifier on the TF-IDF
  features
- Naive Bayes suits text classification well because it assumes word
  independence — a reasonable approximation for bag-of-words style
  features — and is fast even on high-dimensional sparse data
- Used the trained model to predict labels for the test set

## Task 5: Evaluate the Model
| Metric | Score |
|---|---|
| Accuracy | **97.78%** |
| Precision (Spam) | 1.00 |
| Recall (Spam) | 0.82 |
| F1-score (Spam) | 0.90 |

**Confusion Matrix:**
|  | Predicted Ham | Predicted Spam |
|---|---|---|
| **Actual Ham** | 903 | 0 |
| **Actual Spam** | 23 | 108 |

See `confusion_matrix.png` for the visual version.

---

## Observations and Conclusions
1. The dataset is imbalanced (ham heavily outnumbers spam), which is
   typical for real-world spam data and is why accuracy alone isn't
   enough — precision/recall on the spam class matter more.
2. Spam messages are noticeably longer on average than ham messages,
   confirming message length as a useful (if indirect) signal.
3. The model achieved **97.78% accuracy**, with **zero false
   positives** — no legitimate message was ever wrongly flagged as
   spam, which is the most important property for a real spam filter.
4. It missed 23 spam messages (classified as ham) — a recall of 82%.
   This trade-off (perfect precision, imperfect recall) is generally
   preferable for spam filters, since users tolerate a missed spam
   message far more than a lost legitimate one.
5. TF-IDF + Naive Bayes is a lightweight, fast, and surprisingly
   effective baseline for text classification tasks.

## Files
- `spam_classification_project.py` — full pipeline, structured by task
- `SMSSpamCollection` — dataset
- `eda_class_distribution.png` — class balance chart
- `eda_message_length.png` — message length distribution by class
- `confusion_matrix.png` — confusion matrix of the trained model

## How to Run
```bash
pip install scikit-learn pandas nltk matplotlib seaborn
python spam_classification_project.py
```
