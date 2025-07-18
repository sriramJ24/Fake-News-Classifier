# Step 1: Import Libraries
import pandas as pd
import numpy as np
import string
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load and Label the Data
# ASSUMES Fake.csv and True.csv are in a 'data' subdirectory
df_fake = pd.read_csv('./data/Fake.csv')
df_true = pd.read_csv('./data/True.csv')

df_fake['label'] = 'FAKE'
df_true['label'] = 'REAL'

# Step 3: Combine and Shuffle
df = pd.concat([df_fake, df_true], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Step 4: Basic Cleaning Function
def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'\[.*?\\]', '', text)  # remove text in brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove URLs
    text = re.sub(r'<.*?>+', '', text)  # remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # remove punctuation
    text = re.sub(r'\n', ' ', text)  # remove line breaks
    text = re.sub(r'\w*\d\w*', '', text)  # remove words with numbers
    return text.strip()

# Step 5: Apply Cleaning
df['text'] = df['title'] + " " + df['text']  # combine title and text
df['text'] = df['text'].apply(clean_text)

# Step 6: Feature and Label Prep
X = df['text']
y = df['label']

# Step 7: Vectorize the Text (using N-grams)
# Added ngram_range=(1, 2) to include bigrams
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1, 2))
X_vectorized = vectorizer.fit_transform(X)

# Step 8: Split into Train and Test
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# Step 9.1: Initialize and Train the Classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 9.2: Make Predictions
y_pred = model.predict(X_test)

# Step 9.3: Evaluate the Model
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", round(acc * 100, 2), "%")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

# Step 10: Save model and vectorizer
with open('fake_news_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("\nModel and vectorizer saved successfully.") 