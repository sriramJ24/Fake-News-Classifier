import streamlit as st
import pickle
import re
import string

# Load model and vectorizer
try:
    with open('fake_news_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please run train_and_save.py first.")
    st.stop()

# Basic Cleaning Function (must match the one used in training)
def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'\[.*?\\]', '', text)  # remove text in brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove URLs
    text = re.sub(r'<.*?>+', '', text)  # remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # remove punctuation
    text = re.sub(r'\n', ' ', text)  # remove line breaks
    text = re.sub(r'\w*\d\w*', '', text)  # remove words with numbers
    return text.strip()

# -- UI Structure and Content --

# Main Title Section
st.title("Spot Fake News Instantly with AI")
st.write("Protect yourself and loved ones from misinformation. Our AI-powered tool analyzes news articles and headlines to detect fake content in seconds.")

# Add a button like in the screenshot (optional, could link to demo or just be visual)
# st.button("Try Demo Now") # This would just be a visual button

# Why This Matters Section
st.header("Why This Matters")
st.write("Misinformation has become alarmingly common in today's digital age. False news spreads faster than truth on social media, reaching millions of people within hours. This can lead to public panic, election interference, health misinformation, and erosion of trust in legitimate institutions.")
st.write("Vulnerable populations, especially older adults who may be less familiar with digital literacy, are particularly susceptible to sophisticated fake news that uses deepfakes, AI-generated content, and emotional manipulation tactics.")
st.write("Our mission is to democratize access to fact-checking technology, empowering everyone with the tools to identify misinformation before it causes harm to individuals and communities.")

# Live Demo Section (Existing functionality)
st.header("Live Demo")
st.subheader("Test Any News Article or Headline")
st.write("Paste any news content below and our AI will analyze it for you")

user_input = st.text_area("News Text", height=300, label_visibility="collapsed") # hide label for cleaner look

if st.button("Analyze Content"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned_input = clean_text(user_input)
        text_vec = vectorizer.transform([cleaned_input])
        prediction = model.predict(text_vec)[0]
        
        # Display result - simple text for now, cannot replicate card + confidence easily
        if prediction == 'FAKE':
            st.error("Classification: Fake")
        else:
            st.success("Classification: Real")

# How It Works Section
st.header("How It Works")

# Using columns to simulate the 3 steps layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1")
    st.write("**Paste**")
    st.write("Copy and paste any news headline or article content into our analyzer")

with col2:
    st.subheader("2")
    st.write("**Analyze**")
    st.write("Our AI processes the text using advanced machine learning algorithms")

with col3:
    st.subheader("3")
    st.write("**Detect**")
    st.write("Get instant results showing whether the content is likely fake or real")

# The Tech Behind the App Section
st.header("The Tech Behind the App")

# Using columns for the model description and features
col_model, col_features = st.columns([2, 1]) # Adjust column width ratio if needed

with col_model:
    st.subheader("Machine Learning Model")
    st.write("Our classifier uses a **MultinomialNB** combined with **TF-IDF vectorization** to analyze text patterns and linguistic features that distinguish real news from fake news.") # Updated model name
    st.write("The model was trained on thousands of verified real and fake news articles to achieve high accuracy in detecting misinformation.")

with col_features:
    st.write("**\u2022** High Accuracy Detection") # Using unicode for bullet points
    st.write("**\u2022** Real-time Analysis")
    st.write("**\u2022** Advanced NLP Techniques")

# Footer like section (simple text)
st.markdown("---") # Add a horizontal rule for separation
st.write("Built with ❤️ to fight misinformation and protect our communities") 