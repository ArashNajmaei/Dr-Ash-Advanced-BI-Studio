# Dr Ash Multimodel Classifier Studio

An interactive, no-code analytics workbench for applied business research and
teaching. Three modules in one app:

1. **Classification** — churn, credit risk, loan default, conversion, fraud.
   Logistic Regression · Random Forest · Gradient Boosting · Decision Tree · XGBoost.
   Accuracy, precision, recall, F1, ROC-AUC, MCC, confusion matrix, ROC / PR curves,
   feature importance, live threshold tuning, optional 5-fold cross-validation.
2. **Time series** — Exponential Smoothing (Holt-Winters) · ARIMA / SARIMA · LSTM.
   Hold-out backtest metrics (MAE, RMSE, MAPE) plus a forward forecast with a
   confidence band for ARIMA.
3. **Text / Web / Social analytics** — n-gram frequency + word cloud, VADER
   sentiment, LDA topic modelling, TF-IDF keywords. Input from a CSV column,
   pasted text, or a fetched URL.

Each module: load data → pick a method → set parameters → run → visualise.

## Run locally
```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run classifier_studio.py
```
Open http://localhost:8501. Each tab has a demo dataset toggle, or use the sample
CSVs in this repo.

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. **Create app → Deploy from GitHub**, select the repo, set the main file to
   `classifier_studio.py`, and **Deploy**.
4. `packages.txt` installs `libgomp1` so XGBoost works.

## Optional: LSTM
The LSTM forecaster needs TensorFlow, which is large and can exceed the free
Streamlit tier's memory. To enable it, uncomment the `tensorflow-cpu` line in
`requirements.txt` and redeploy. Without it, the LSTM option shows a friendly
notice and the other forecasters work normally.

## Files
| File | Purpose |
|------|---------|
| `classifier_studio.py` | The Streamlit app (three tabs) |
| `requirements.txt` | Python dependencies |
| `packages.txt` | System dependency (`libgomp1`) for XGBoost |
| `sample_churn.csv` | Demo data for the Classification tab |
| `sample_sales.csv` | Demo data for the Time-series tab |
| `sample_reviews.csv` | Demo data for the Text analytics tab |
