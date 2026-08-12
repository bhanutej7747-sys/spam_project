"""
Project 4: Spam Email Classification
-------------------------------------
Objective:
Build a machine learning model to classify SMS messages as Spam or Ham.

Dataset:
SMS Spam Collection Dataset

Tasks:
1. Load, clean, and preprocess the dataset.
2. Convert the text data into numerical features and perform basic EDA.
3. Split the dataset into training and testing sets.
4. Train a Naive Bayes classifier and predict the test data.
5. Evaluate the model using Accuracy Score and Confusion Matrix, then write conclusions.
"""

import re
import string
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

RANDOM_STATE = 42


# ===================================================================
# TASK 1: LOAD, CLEAN AND PREPROCESS THE DATASET
# ===================================================================
print("=" * 60)
print("TASK 1: Load, Clean and Preprocess the Dataset")
print("=" * 60)

df = pd.read_csv("SMSSpamCollection", sep="\t", header=None,
                  names=["label", "message"])

print(f"\nShape of dataset: {df.shape}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nDuplicate rows: {df.duplicated().sum()}")

# Remove duplicates
df = df.drop_duplicates().reset_index(drop=True)
print(f"\nShape after removing duplicates: {df.shape}")

print(f"\nClass distribution:\n{df['label'].value_counts()}")

# Text cleaning function
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def clean_text(text):
    """Lowercase, remove URLs/punctuation/digits, remove stopwords, stem."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 1]
    return " ".join(tokens)


df["clean_message"] = df["message"].apply(clean_text)
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
df["message_length"] = df["message"].apply(len)

print("\nSample of cleaned data:")
print(df[["label", "message", "clean_message"]].head())


# ===================================================================
# TASK 2: CONVERT TEXT TO NUMERICAL FEATURES + BASIC EDA
# ===================================================================
print("\n" + "=" * 60)
print("TASK 2: Feature Conversion (TF-IDF) and Basic EDA")
print("=" * 60)

# --- EDA Plot 1: Class distribution ---
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="label", hue="label", palette="Set2", legend=False)
plt.title("Class Distribution: Ham vs Spam")
plt.xlabel("Label")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("eda_class_distribution.png", dpi=150)
plt.close()

# --- EDA Plot 2: Message length distribution by class ---
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x="message_length", hue="label", bins=40,
             kde=True, palette="Set2", element="step")
plt.title("Message Length Distribution by Class")
plt.xlabel("Message Length (characters)")
plt.tight_layout()
plt.savefig("eda_message_length.png", dpi=150)
plt.close()

print("\nAverage message length by class:")
print(df.groupby("label")["message_length"].mean())
print("\nSaved EDA plots: eda_class_distribution.png, eda_message_length.png")

# --- Convert text to numerical features using TF-IDF ---
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df["clean_message"])
y = df["label_num"]

print(f"\nTF-IDF feature matrix shape: {X.shape}")


# ===================================================================
# TASK 3: TRAIN-TEST SPLIT
# ===================================================================
print("\n" + "=" * 60)
print("TASK 3: Train-Test Split")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")


# ===================================================================
# TASK 4: TRAIN NAIVE BAYES CLASSIFIER
# ===================================================================
print("\n" + "=" * 60)
print("TASK 4: Train Naive Bayes Classifier")
print("=" * 60)

model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nModel trained successfully.")
print("\nSample predictions (first 10 test messages):")
for actual, pred in list(zip(y_test[:10], y_pred[:10])):
    label = lambda x: "Spam" if x == 1 else "Ham"
    print(f"Actual: {label(actual):5s} | Predicted: {label(pred)}")


# ===================================================================
# TASK 5: EVALUATE THE MODEL
# ===================================================================
print("\n" + "=" * 60)
print("TASK 5: Evaluate the Model")
print("=" * 60)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\nAccuracy Score: {acc:.4f} ({acc*100:.2f}%)")
print(f"\nConfusion Matrix:\n{cm}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])}")

# Plot confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.title(f"Confusion Matrix - Naive Bayes (Accuracy: {acc*100:.2f}%)")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("\nSaved confusion_matrix.png")

# ===================================================================
# OBSERVATIONS AND CONCLUSIONS
# ===================================================================
print("\n" + "=" * 60)
print("OBSERVATIONS AND CONCLUSIONS")
print("=" * 60)
tn, fp, fn, tp = cm.ravel()
print(f"""
1. The dataset is imbalanced: {df['label'].value_counts()['ham']} ham vs
   {df['label'].value_counts()['spam']} spam messages. Ham messages dominate,
   which is typical for real-world spam datasets.

2. Spam messages tend to be longer on average than ham messages, since they
   often contain promotional text, links, and call-to-action phrases.

3. The Multinomial Naive Bayes model achieved an accuracy of {acc*100:.2f}%
   on unseen test data, which is strong for a simple probabilistic model
   and confirms TF-IDF features capture spam-indicating word patterns well.

4. Confusion matrix breakdown: {tn} true negatives (ham correctly
   identified), {tp} true positives (spam correctly identified), {fp} false
   positives (ham wrongly flagged as spam), {fn} false negatives (spam
   missed and passed as ham).

5. Naive Bayes works particularly well here because it assumes word
   independence, which is a reasonable approximation for bag-of-words /
   TF-IDF style text features, and it is computationally efficient even on
   sparse high-dimensional text data.
""")
