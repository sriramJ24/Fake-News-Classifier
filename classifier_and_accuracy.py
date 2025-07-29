import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Basic text cleaning - just lowercase
def text_cleaning(text):
    return text.lower()

# Load data
df_fake = pd.read_csv('./data/Fake.csv')
df_true = pd.read_csv('./data/True.csv')

df_fake['label'] = 'FAKE'
df_true['label'] = 'REAL'

# Combine data
df = pd.concat([df_fake, df_true], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Basic text preparation
df['text'] = df['title'] + " " + df['text']
df['text'] = df['text'].apply(text_cleaning)

X = df['text']
y = df['label']

# Basic vectorizer - just counts words
vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# ===== ACCURACY TEST SECTION (TERMINAL ONLY) =====
print("=" * 50)
print("ACCURACY TEST RESULTS")
print("=" * 50)

# Make predictions on test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {accuracy:.2%}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print("                  Predicted")
print("                  FAKE  REAL")
print(f"Actual FAKE:     {cm[0][0]:4d}  {cm[0][1]:4d}")
print(f"Actual REAL:     {cm[1][0]:4d}  {cm[1][1]:4d}")

# Detailed Classification Report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# Calculate additional metrics
total_samples = len(y_test)
correct_predictions = (y_test == y_pred).sum()
incorrect_predictions = total_samples - correct_predictions

print(f"\nSummary:")
print(f"Total test samples: {total_samples}")
print(f"Correct predictions: {correct_predictions}")
print(f"Incorrect predictions: {incorrect_predictions}")
print(f"Accuracy: {accuracy:.2%}")

# Per-class accuracy
fake_accuracy = cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
real_accuracy = cm[1][1] / (cm[1][0] + cm[1][1]) if (cm[1][0] + cm[1][1]) > 0 else 0

print(f"FAKE news detection accuracy: {fake_accuracy:.2%}")
print(f"REAL news detection accuracy: {real_accuracy:.2%}")

print("=" * 50)

# Save model
with open('basic_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('basic_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Basic model saved!")

# Streamlit app
st.title("🎯 Fake News Classifier")
st.write("A simple machine learning approach to detect fake news.")

st.header("How it works:")
st.write("""
1. **Text Cleaning**: Converts text to lowercase
2. **Word Counting**: Counts how many times each word appears
3. **Learning**: Uses Naive Bayes to learn patterns
4. **Prediction**: Classifies new articles as FAKE or REAL
""")

st.header("Test an Article")
user_input = st.text_area("Paste any news article or headline here:", height=200)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Clean the text
        cleaned_text = text_cleaning(user_input)
        
        # Convert to numbers
        text_vector = vectorizer.transform([cleaned_text])
        
        # Make prediction
        prediction = model.predict(text_vector)[0]
        
        # Get prediction probabilities
        probabilities = model.predict_proba(text_vector)[0]
        confidence = max(probabilities) * 100
        
        # Show result
        if prediction == 'FAKE':
            st.error(f"🔴 Prediction: FAKE (Confidence: {confidence:.1f}%)")
        else:
            st.success(f"🟢 Prediction: REAL (Confidence: {confidence:.1f}%)")
        
        # Show what words were found
        st.write("**Words the model found in your text:**")
        words_found = []
        for word in cleaned_text.split():
            if word in vectorizer.vocabulary_:
                words_found.append(word)
        
        if words_found:
            st.write(", ".join(words_found[:20]))
        else:
            st.write("No recognized words found.")
        
        # Show debugging info
        st.write("**Debugging Information:**")
        st.write(f"FAKE probability: {probabilities[0]:.2%}")
        st.write(f"REAL probability: {probabilities[1]:.2%}")
        

st.header("Try These Examples:")
st.write("""
**Fake News Example:**
"BREAKING: SHOCKING conspiracy theory about 5G towers!"

**Real News Example:**
"Study shows benefits of exercise for mental health."
""")

st.markdown("---")
st.write("This is still very much a work in progress, and I'm still learning how to use Streamlit.")
