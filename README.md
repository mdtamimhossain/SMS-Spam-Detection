SMS Spam Detection — Simple README

Overview
This notebook implements a simple SMS spam detector using:
- Hand-built Bag-of-Words and TF–IDF feature extractors (from scratch).
- A baseline classifier: Multinomial Naive Bayes (sklearn).

Files
- `Assignment_2.ipynb`: Main notebook with code, tests, and a short report.
- `Data sources/SMSSpamCollection`: SMS dataset (tab-separated, label \t message).

Setup (minimal)
Run these commands in your Python environment:

```bash
pip install pandas numpy scipy scikit-learn ipykernel
```

How to run
1. Open `Assignment_2.ipynb` in Jupyter or VS Code Notebook.
2. Select the Python kernel (the one with the installed packages).
3. Run cells from top to bottom. The notebook includes small test functions that run automatically in the test cell.

Quick single-SMS test (example)
Inside the notebook you can run the example near the end:

```python
sample_sms = "free laptop click this"
sample_tokens = clean_text(sample_sms)
sample_features, _ = bag_of_words([sample_tokens], vocab)
print('tokens:', sample_tokens)
print('features shape:', sample_features.shape)
print('prediction (1=spam,0=ham):', model.predict(sample_features))
```

What the tests check
- `load_sms_dataset`: data loads and has `label` and `message` columns.
- `clean_text`: returns a token list.
- `build_vocabulary`, `bag_of_words`, `tf_idf`: basic shape and value checks.
- `train_naive_bayes` and `evaluate_model`: basic train and metric print.

Findings (short)
- Naive Bayes is a good baseline for this dataset.
- BoW and TF–IDF give similar baseline performance here.

Limitations
- Very basic preprocessing; no stopwords, stemming, or URL/number handling.
- No cross-validation or hyperparameter search by default.

Next steps
- Improve preprocessing (normalize URLs/numbers, add stemming).
- Limit vocabulary (min document frequency) or use feature selection.
- Try other classifiers and use cross-validation for more reliable scores.
