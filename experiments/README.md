# Generalization experiments

These scripts chose the model settings without ever tuning on the McIntire test set.

## How settings were chosen

- **A** = ISOT. **B** = WELFake with every article that matches ISOT or McIntire removed (about 17,500 articles, mostly the 2016 Kaggle set). WELFake was partly built from McIntire, so this removal matters.
- **Selection score:** train on A and test on B, then train on B and test on A, and average the balanced accuracy (the mean of fake recall and real recall).
- **More sources:** B was split by outlet (New York Times real articles in one half, Breitbart and others in the other). Adding one half to ISOT was then tested on the other half's unseen outlets.
- McIntire was used once, at the end (`final_test.py`), after the setup was fixed.

## Run order (from the repo root)

```bash
.venv/bin/python experiments/overlap.py                     # builds welfake_residual.pkl (B)
.venv/bin/python experiments/run_experiments.py base_v1_C10 "v2_tfidf+stylo_C1"   # any names in CONFIGS
.venv/bin/python experiments/multi_exp.py "v2_tfidf+stylo_C1"
.venv/bin/python experiments/bert_exp.py --cleaner v2 --out bert_A   # ~25 min on an M2
.venv/bin/python experiments/bert_ensemble_val.py bert_A
.venv/bin/python experiments/final_test.py                  # the one McIntire test
```

Results: `results.jsonl` / `batch1.log` (validation), `multi.log` (more sources), `final_test.log` (McIntire).

Note: `run_experiments.py` computes style features with pandas, whose string engine handles a few
unusual characters slightly differently from Python's `re`. Every McIntire number comes from the production
code in `text_utils.py`, so this only affects the validation figures.
