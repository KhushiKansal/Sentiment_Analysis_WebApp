# # train_model.py

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# import joblib

# # 1. Load dataset
# df = pd.read_csv("twitter_sentiment.csv")

# # Rename columns if needed (depends on dataset)
# # The airline dataset usually has 'text' and 'airline_sentiment'
# texts = df["text"].astype(str)
# labels = df["airline_sentiment"]

# # 2. Split into train/test
# X_train, X_test, y_train, y_test = train_test_split(
#     texts, labels, test_size=0.2, random_state=42
# )

# # 3. Convert text to numbers using TF-IDF
# vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
# X_train_tfidf = vectorizer.fit_transform(X_train)
# X_test_tfidf = vectorizer.transform(X_test)

# # 4. Train a simple classifier (Logistic Regression)
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train_tfidf, y_train)

# # 5. Evaluate
# y_pred = model.predict(X_test_tfidf)
# print("Accuracy:", accuracy_score(y_test, y_pred))

# # 6. Save model + vectorizer
# joblib.dump(model, "sentiment_model.pkl")
# joblib.dump(vectorizer, "vectorizer.pkl")

# print("Model and vectorizer saved successfully!")

# train_model.py

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Download NLTK stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# 1. Load dataset
df = pd.read_csv("twitter_sentiment.csv")

# 2. Balance classes safely
min_count = df['airline_sentiment'].value_counts().min()

neg = df[df.airline_sentiment == "negative"].sample(min_count, random_state=42)
pos = df[df.airline_sentiment == "positive"].sample(min_count, random_state=42)
neu = df[df.airline_sentiment == "neutral"].sample(min_count, random_state=42)

df_balanced = pd.concat([neg, pos, neu]).sample(frac=1, random_state=42)  # shuffle

texts = df_balanced["text"].astype(str)
labels = df_balanced["airline_sentiment"]

# 3. Preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)  # remove punctuation/numbers
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

texts_cleaned = texts.apply(clean_text)

# 4. Split
X_train, X_test, y_train, y_test = train_test_split(
    texts_cleaned, labels, test_size=0.2, random_state=42
)

# 5. TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 6. Train Logistic Regression
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_tfidf, y_train)

# 7. Evaluate
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 8. Save model + vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Balanced model with NLTK preprocessing saved successfully!")

