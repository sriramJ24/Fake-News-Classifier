# The Daily Check: Fake News Classifier

A small web app that estimates whether a news article is real or fake. Paste an article or a link, and it shows a verdict, the words that drove the decision, and (optionally) a fact-check of the claims against the web.

## Why I built it

I created a Fake News Classifier after my grandfather from India unknowingly sent me an article that was clearly fake and written by AI. It made me realize how easy it is for older people to be misled, especially with AI becoming more and more prevalent in media nowadays. So the reason I built this was to help people tell the difference between real and fake news and be able to make the right decisions.

## How it works

- **Model:** two kinds of features feed one Logistic Regression classifier. TF-IDF covers *what* the article says (single words and two-word phrases). 17 writing-style features cover *how* it is written: exclamation marks, questions, ALL-CAPS words, talking to "you", quoting and attributing sources ("said", "told"), sentence length, and so on. It is trained on about 50,000 unique articles from the [ISOT](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/) and [WELFake](https://zenodo.org/records/4561253) datasets, with each dataset given equal weight so the bigger one doesn't set the style.
- **Cleaning:** It strips giveaways that reveal the *source* rather than the content: datelines like "WASHINGTON (Reuters) -", outlet names tacked onto headlines (" - The New York Times"), bylines, "Follow X on Twitter", "Featured image via Getty", URLs, @handles, and curly-vs-straight quotes. Without this step the model mostly learns "Reuters = real".
- **Fair testing:** Duplicate and empty articles are removed, and the data is split *before* the vectorizer is fitted, so test articles are truly unseen.
- **Results on the same datasets:** 96.3% accuracy on 12,545 held-out articles (ISOT 98.5%, WELFake 92.7%). Headlines alone get only about 71%, so the app won't give a verdict on very short text.
- **Results on a new dataset:** This is the more honest number. The model is tested on 6,202 articles from a separate dataset ([McIntire](https://github.com/lutzhamel/fake-news/tree/master/data)) after removing every training article that matches one of them. WELFake was partly built from McIntire, so this step matters. It scores **82.7%** (fake caught 93%, real kept 73%), up from 58.9% before these changes. On its own training datasets the same setup scores 96.7% (5-fold cross-validation). Run `python evaluate_generalization.py` to reproduce this.
- **Optional DistilBERT:** a fine-tuned transformer model. Trained on ISOT alone it scored 67.3% on McIntire, and averaging it with the main model *lowered* accuracy to 72.0%, so the app no longer uses it unless you set `USE_DISTILBERT=1`.
- **Optional fact-check:** If an Anthropic API key is set, a "Fact-check the claims online" button asks Claude to search the web and check the article's main claims.

### What improved accuracy on the new dataset

Settings were chosen on a validation setup that never touches McIntire: train on ISOT and test on the non-McIntire part of WELFake, and the reverse. McIntire was tested once at the end. Full details and scripts are in [`experiments/`](experiments/README.md).

| Approach | Same-dataset accuracy | McIntire accuracy | Fake recall | Real recall |
|---|---|---|---|---|
| Before: words only, trained on ISOT | 98.9% | 58.9% | 90% | 28% |
| Words + style, trained on ISOT | 98.8% | 71.8% | 64% | 79% |
| Words + style, ISOT + WELFake | 96.4% | 81.5% | 91% | 73% |
| **Words + style, ISOT + WELFake, equal weights (current)** | **96.6%** | **82.7%** | **93%** | **73%** |
| DistilBERT, trained on ISOT | 99.6% | 67.3% | 76% | 59% |
| DistilBERT + words/style (ISOT), averaged | n/a | 70.6% | 74% | 67% |
| DistilBERT + current model, averaged | n/a | 72.0% | 82% | 62% |

Same-dataset accuracy here is an 80/20 split of each approach's own training data. Several other ideas were tested on validation and did not help, so they were not used: stronger regularization, unigrams only, a smaller vocabulary, binary word features, character n-grams, and also removing weekday names. The biggest gains came from the style features, which point the same way in every dataset, and from training on more than one source.

**Limits:** The classifier judges *writing style*, not truth. On a dataset it never saw it is right about 4 times in 5, and it still calls about a quarter of real articles fake. Both datasets are mostly 2015–2018 English-language politics, so it is less reliable on other topics and on newer articles, including AI-written ones. Use the fact-check and a trusted news site for anything important.

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
4. Optional: add `fake_or_real_news.csv` (McIntire) to `data/` and run `python evaluate_generalization.py`. It adds the new-dataset result to `metrics.json`.

### Optional: DistilBERT model

```bash
python3 -m pip install -r requirements-transformer.txt
python train_transformer.py          # about 25 minutes on an M2 Mac; --sample sets the training size
```

This saves the model to `models/distilbert/`. Start the app with `USE_DISTILBERT=1 streamlit run app.py` to average it in. On unseen articles this made results worse (see above).

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
| `evaluate_generalization.py` | Tests the model on a dataset it never trained on |
| `text_utils.py` | Text cleaning and writing-style features, shared by training and the app |
| `experiments/` | The validation experiments behind the results table |
| `model.pkl`, `metrics.json` | Trained model and its test results |
| `assets/newspaper_bg.jpg` | Background image |
