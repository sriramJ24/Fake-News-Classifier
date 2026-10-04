# The Daily Check: Fake News Classifier

A small web app that estimates whether a news article is real or fake. Paste an article or a link, and it shows a verdict, the words that drove the decision, and (optionally) a fact-check of the claims against the web.

## Why I built it

I created a Fake News Classifier after my grandfather from India unknowingly sent me an article that was clearly fake and written by AI. It made me realize how easy it is for older people to be misled, especially with AI becoming more and more prevalent in media nowadays. So the reason I built this was to help people tell the difference between real and fake news and be able to make the right decisions.

## How it works

- **Model:** TF-IDF features (single words and two-word phrases) with Logistic Regression, trained on about 50,000 unique articles from the [ISOT](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/) and [WELFake](https://zenodo.org/records/4561253) datasets.
- **Cleaning:** It strips giveaways that reveal the *source* rather than the content, like the "WASHINGTON (Reuters) -" opening on every real ISOT article, "Featured image via Getty", URLs and @handles. Without this step the model mostly learns "Reuters = real".
- **Fair testing:** Duplicate and empty articles are removed, and the data is split *before* the vectorizer is fitted, so test articles are truly unseen.
- **Results:** 96.7% accuracy on 12,547 unseen articles (ISOT 98.3%, WELFake 94.1%). Headlines alone get only about 71%, so the app won't give a verdict on very short text.
- **Optional DistilBERT:** a fine-tuned transformer model that reads meaning rather than single words. If it has been trained, the app averages its prediction with the main model's. On 4,000 unseen articles: word model 96.8%, DistilBERT 98.6%, **both together 99.2%**.
- **Optional fact-check:** If an Anthropic API key is set, a "Fact-check the claims online" button asks Claude to search the web and check the article's main claims.

**Limits:** The classifier judges *writing style*, not truth. Both datasets are mostly 2015–2018 English-language politics, so it is less reliable on other topics and on newer articles, including AI-written ones. Use the fact-check and a trusted news site for anything important.

## How to run

```bash
python3 -m pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. The trained `model.pkl` is included, so no data is needed just to run the app.

### Retrain the model

1. Put `Fake.csv` and `True.csv` (ISOT) in `data/`.
2. Optional: add `WELFake_Dataset.csv` from Zenodo to `data/`.
3. Run `python train.py`. It prints test results and saves `model.pkl` and `metrics.json`.

### Optional: DistilBERT model

```bash
python3 -m pip install -r requirements-transformer.txt
python train_transformer.py          # about 25 minutes on an M2 Mac; --sample sets the training size
```

This saves the model to `models/distilbert/`. The app picks it up automatically.

### Optional: fact-check with Claude

Set `ANTHROPIC_API_KEY` in your environment, or add it to `.streamlit/secrets.toml`:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

Each fact-check runs a web search and costs a few cents.

## Files

| File | What it does |
|---|---|
| `app.py` | Streamlit web app |
| `train.py` | Cleans data, trains and evaluates the model |
| `train_transformer.py` | Optional DistilBERT fine-tuning |
| `text_utils.py` | Text cleaning shared by training and the app |
| `model.pkl`, `metrics.json` | Trained model and its test results |
| `assets/newspaper_bg.jpg` | Background image |
