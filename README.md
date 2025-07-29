# Fake News Classifier

A simple machine learning model that classifies news articles as FAKE or REAL.

## What it does

- Analyzes text using word patterns
- Classifies articles as FAKE or REAL using Naive Bayes Classifier
- Shows confidence level for predictions
- Displays debugging information

## How to run

### 1. Install dependencies
```bash
python3 -m pip install pandas scikit-learn streamlit
```

### 2. Prepare data
- Create a `data/` folder
- Put `Fake.csv` and `True.csv` in the `data/` folder

### 3. Run the app
```bash
streamlit run classifier_and_accuracy.py
or
python3 -m streamlit run classifier_and_accuracy.py
```

The app will:
- Train the model automatically
- Show accuracy results in terminal
- Open web interface in browser

## Usage

1. Paste any news article in the text box
2. Click "Analyze"
3. See the classification result

## Purpose of the project

I created a Fake News Classifier after my grandfather from India unknowingly sent me an article that was clearly fake and written by AI. It made me realize how easy it is for older people to be misled, especially with AI becoming more and more prevalent in media nowadays. So the reason I built this was to help people tell the difference between real and fake news and be able to make the right decisions.
