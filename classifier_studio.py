"""
Dr Ash Multimodel Classifier Studio
===================================
An interactive, no-code analytics workbench for applied business research and
teaching, with three modules:

  1. Classification   — churn, credit risk, loan default, etc.
                        Logistic Regression, Random Forest, Gradient Boosting,
                        Decision Tree, XGBoost. Full metric + visualisation suite.

  2. Time series      — Exponential Smoothing (Holt-Winters), ARIMA / SARIMA,
                        and LSTM. Backtest metrics + forward forecast.

  3. Text / Web /     — n-gram frequency + word cloud, VADER sentiment,
     Social analytics    LDA topic modelling, TF-IDF keywords. Input from a CSV
                         column, pasted text, or a fetched URL.

Run locally:
    pip install -r requirements.txt
    streamlit run classifier_studio.py

LSTM needs TensorFlow, which is heavy. It is optional: uncomment `tensorflow-cpu`
in requirements.txt to enable it. The app runs fine without it.
"""

import io
import re

import numpy as np
import pandas as pd

# ---- classification ------------------------------------------------------- #
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    average_precision_score, auc,
)

# ---- text ----------------------------------------------------------------- #
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS)
from sklearn.decomposition import LatentDirichletAllocation

# ---- time series ---------------------------------------------------------- #
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

# ---- optional heavy deps -------------------------------------------------- #
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import tensorflow as tf  # noqa: F401
    HAS_TF = True
except Exception:
    HAS_TF = False

# --------------------------------------------------------------------------- #
# Brand palette
# --------------------------------------------------------------------------- #
NAVY = "#0F2138"
BRASS = "#C29B4A"
NAVY_SOFT = "#33465E"
GREY = "#8A94A6"
BG_TINT = "#F5F3EE"

MODEL_NAMES = [
    "Logistic Regression", "Random Forest", "Gradient Boosting",
    "Decision Tree", "XGBoost",
]

FREQ_MAP = {
    "Infer": (None, 0),
    "Daily (D)": ("D", 7),
    "Weekly (W)": ("W", 52),
    "Monthly (MS)": ("MS", 12),
    "Quarterly (QS)": ("QS", 4),
    "Yearly (YS)": ("YS", 1),
}

# =========================================================================== #
# CLASSIFICATION — pure helpers
# =========================================================================== #

def infer_column_types(df, columns):
    numeric, categorical = [], []
    for c in columns:
        (numeric if pd.api.types.is_numeric_dtype(df[c]) else categorical).append(c)
    return numeric, categorical


def suggest_targets(df, max_unique=15):
    return [c for c in df.columns if 2 <= df[c].nunique(dropna=True) <= max_unique]


def build_preprocessor(numeric_cols, categorical_cols, scale_numeric=True):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(num_steps)
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", cat_pipe, categorical_cols))
    return ColumnTransformer(transformers, remainder="drop")


def encode_target(y, positive_label=None):
    classes = sorted(pd.Series(y).dropna().unique().tolist(), key=lambda v: str(v))
    if len(classes) == 2 and positive_label is not None:
        neg = [c for c in classes if c != positive_label][0]
        mapping, ordered = {neg: 0, positive_label: 1}, [neg, positive_label]
    else:
        mapping = {c: i for i, c in enumerate(classes)}
        ordered = classes
    return pd.Series(y).map(mapping).to_numpy(), ordered


def get_model(name, params, n_classes, use_imbalance):
    cw = "balanced" if use_imbalance else None
    seed = params.get("seed", 42)
    if name == "Logistic Regression":
        return LogisticRegression(C=params.get("C", 1.0), max_iter=2000, class_weight=cw)
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth"), class_weight=cw,
            random_state=seed, n_jobs=-1)
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 3),
            learning_rate=params.get("learning_rate", 0.1), random_state=seed)
    if name == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=params.get("max_depth"), class_weight=cw, random_state=seed)
    if name == "XGBoost":
        if not HAS_XGB:
            raise RuntimeError("xgboost is not installed.")
        kwargs = dict(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", 4),
            learning_rate=params.get("learning_rate", 0.1),
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=seed, n_jobs=-1, tree_method="hist")
        if n_classes == 2:
            kwargs["scale_pos_weight"] = (
                params.get("scale_pos_weight", 1) if use_imbalance else 1)
        return XGBClassifier(**kwargs)
    raise ValueError(f"Unknown model: {name}")


def compute_metrics(y_true, y_pred, y_proba, n_classes):
    avg = "binary" if n_classes == 2 else "macro"
    m = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    try:
        m["ROC-AUC"] = (roc_auc_score(y_true, y_proba[:, 1]) if n_classes == 2
                        else roc_auc_score(y_true, y_proba, multi_class="ovr",
                                           average="macro"))
    except Exception:
        m["ROC-AUC"] = float("nan")
    return m


def get_importances(model, feature_names):
    if hasattr(model, "feature_importances_"):
        return pd.DataFrame({"feature": feature_names,
                             "importance": np.asarray(model.feature_importances_)})
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if coef.ndim == 2 and coef.shape[0] == 1:
            vals = coef.ravel()
        elif coef.ndim == 1:
            vals = coef
        else:
            vals = np.mean(np.abs(coef), axis=0)
        return pd.DataFrame({"feature": feature_names, "importance": vals})
    return None


def run_classification(X, y_encoded, n_classes, model_name, params, test_size,
                       scale_numeric, use_imbalance, seed):
    numeric_cols, categorical_cols = infer_column_types(X, list(X.columns))
    pre = build_preprocessor(numeric_cols, categorical_cols, scale_numeric)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=seed, stratify=y_encoded)
    model = get_model(model_name, {**params, "seed": seed}, n_classes, use_imbalance)
    clf = Pipeline([("pre", pre), ("model", model)]).fit(X_train, y_train)
    feature_names = list(clf.named_steps["pre"].get_feature_names_out())
    return {
        "clf": clf, "y_test": y_test, "y_proba": clf.predict_proba(X_test),
        "importances": get_importances(clf.named_steps["model"], feature_names),
        "X_test": X_test, "n_classes": n_classes,
    }


def cross_validate_classification(X, y_encoded, n_classes, model_name, params,
                                  scale_numeric, use_imbalance, seed):
    numeric_cols, categorical_cols = infer_column_types(X, list(X.columns))
    pre = build_preprocessor(numeric_cols, categorical_cols, scale_numeric)
    model = get_model(model_name, {**params, "seed": seed}, n_classes, use_imbalance)
    clf = Pipeline([("pre", pre), ("model", model)])
    scoring = (["accuracy", "f1_macro", "roc_auc_ovr"] if n_classes > 2
               else ["accuracy", "f1", "roc_auc"])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    out = cross_validate(clf, X, y_encoded, cv=cv, scoring=scoring)
    return {k: (float(np.mean(v)), float(np.std(v)))
            for k, v in out.items() if k.startswith("test_")}


# =========================================================================== #
# TIME SERIES — pure helpers
# =========================================================================== #

def prepare_series(df, date_col, value_col, freq_key):
    s = df[[date_col, value_col]].copy()
    s[date_col] = pd.to_datetime(s[date_col], errors="coerce")
    s[value_col] = pd.to_numeric(s[value_col], errors="coerce")
    s = s.dropna().sort_values(date_col).set_index(date_col)[value_col]
    s = s[~s.index.duplicated(keep="last")]
    freq, seasonal = FREQ_MAP[freq_key]
    if freq is not None:
        s = s.asfreq(freq)
        if s.isna().any():
            s = s.interpolate().ffill().bfill()
    return s, seasonal


def forecast_metrics(actual, pred):
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    mae = float(np.mean(np.abs(actual - pred)))
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    mask = actual != 0
    mape = (float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)
            if mask.any() else float("nan"))
    return {"MAE": mae, "RMSE": rmse, "MAPE %": mape}


def fit_exp_smoothing(train, horizon, trend, seasonal, seasonal_periods):
    model = ExponentialSmoothing(
        train, trend=trend, seasonal=seasonal,
        seasonal_periods=seasonal_periods if seasonal else None,
        initialization_method="estimated")
    fit = model.fit()
    return fit.forecast(horizon), None


def fit_arima(train, horizon, order, seasonal_order):
    fit = ARIMA(train, order=order, seasonal_order=seasonal_order).fit()
    res = fit.get_forecast(horizon)
    ci = res.conf_int()
    ci.columns = ["lower", "upper"]
    return res.predicted_mean, ci


def auto_arima(train, max_p=3, max_d=2, max_q=3):
    best_order, best_aic = (1, 1, 1), np.inf
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    aic = ARIMA(train, order=(p, d, q)).fit().aic
                    if aic < best_aic:
                        best_order, best_aic = (p, d, q), aic
                except Exception:
                    continue
    return best_order, best_aic


def _make_supervised(values, n_lags):
    X, y = [], []
    for i in range(n_lags, len(values)):
        X.append(values[i - n_lags:i])
        y.append(values[i])
    return np.array(X), np.array(y)


def fit_lstm(train, horizon, n_lags=12, units=32, epochs=30):
    from tensorflow import keras
    vals = train.values.astype("float32")
    vmin, vmax = float(vals.min()), float(vals.max())
    rng = (vmax - vmin) or 1.0
    scaled = (vals - vmin) / rng
    X, y = _make_supervised(scaled, n_lags)
    X = X.reshape((X.shape[0], n_lags, 1))
    model = keras.Sequential([
        keras.layers.Input((n_lags, 1)),
        keras.layers.LSTM(units),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=16, verbose=0)
    window = scaled[-n_lags:].tolist()
    preds = []
    for _ in range(horizon):
        x = np.array(window[-n_lags:]).reshape((1, n_lags, 1))
        p = float(model.predict(x, verbose=0)[0, 0])
        preds.append(p)
        window.append(p)
    return np.array(preds) * rng + vmin


def future_index(series, horizon):
    idx = series.index
    if isinstance(idx, pd.DatetimeIndex) and idx.freq is not None:
        return list(pd.date_range(idx[-1], periods=horizon + 1, freq=idx.freq)[1:])
    if isinstance(idx, pd.DatetimeIndex) and len(idx) > 1:
        step = idx[-1] - idx[-2]
        return [idx[-1] + step * (i + 1) for i in range(horizon)]
    return list(range(len(series), len(series) + horizon))


# =========================================================================== #
# TEXT — pure helpers
# =========================================================================== #

def _stopword_list(use_stopwords, extra):
    sw = list(ENGLISH_STOP_WORDS) if use_stopwords else None
    extra = [w.strip().lower() for w in (extra or []) if w.strip()]
    if extra:
        sw = (sw or []) + extra
    return sw


def word_frequencies(texts, ngram, top_n, use_stopwords, extra):
    cv = CountVectorizer(ngram_range=ngram, stop_words=_stopword_list(use_stopwords, extra))
    Xc = cv.fit_transform(texts)
    counts = np.asarray(Xc.sum(axis=0)).ravel()
    vocab = np.array(cv.get_feature_names_out())
    order = counts.argsort()[::-1][:top_n]
    return pd.DataFrame({"term": vocab[order], "count": counts[order].astype(int)})


def vader_scores(texts):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    rows = [sia.polarity_scores(t) for t in texts]
    df = pd.DataFrame(rows)
    df["label"] = pd.cut(df["compound"], [-1.01, -0.05, 0.05, 1.01],
                         labels=["negative", "neutral", "positive"])
    df["text"] = texts
    return df


def lda_topics(texts, n_topics, n_top, use_stopwords, extra, seed=42):
    min_df = 2 if len(texts) > 4 else 1
    cv = CountVectorizer(stop_words=_stopword_list(use_stopwords, extra),
                         max_df=0.95, min_df=min_df)
    X = cv.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=seed,
                                    learning_method="batch").fit(X)
    vocab = np.array(cv.get_feature_names_out())
    rows = []
    for k, comp in enumerate(lda.components_):
        idx = comp.argsort()[::-1][:n_top]
        rows.append({"Topic": k + 1, "Top terms": ", ".join(vocab[idx])})
    return pd.DataFrame(rows)


def tfidf_keywords(texts, top_n, use_stopwords, extra):
    tv = TfidfVectorizer(stop_words=_stopword_list(use_stopwords, extra))
    X = tv.fit_transform(texts)
    scores = np.asarray(X.mean(axis=0)).ravel()
    vocab = np.array(tv.get_feature_names_out())
    order = scores.argsort()[::-1][:top_n]
    return pd.DataFrame({"term": vocab[order], "tfidf": scores[order].round(4)})


def fetch_url_text(url):
    import requests
    from bs4 import BeautifulSoup
    r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())


def split_sentences(text, min_len=25):
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) >= min_len]


# =========================================================================== #
# Streamlit — shared helpers
# =========================================================================== #

def load_csv(file, st):
    for enc in ("utf-8", "latin-1"):
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except Exception:
            continue
    st.error("Could not parse the CSV — check delimiter and encoding.")
    return None


def demo_churn(n=2000, seed=7):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, n)
    monthly = rng.normal(70, 25, n).clip(15, 150)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n,
                          p=[0.55, 0.25, 0.20])
    calls = rng.poisson(1.5, n)
    senior = rng.integers(0, 2, n)
    logit = (-1.0 - 0.03 * tenure + 0.015 * monthly + 0.35 * calls + 0.4 * senior
             + np.where(contract == "Month-to-month", 1.1, 0.0)
             + rng.normal(0, 0.5, n))
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "tenure_months": tenure, "monthly_charges": monthly.round(2),
        "contract_type": contract, "support_calls": calls,
        "is_senior": senior, "churned": np.where(churn == 1, "Yes", "No")})


def demo_timeseries(seed=0):
    idx = pd.date_range("2019-01-01", periods=72, freq="MS")
    t = np.arange(72)
    vals = (500 + 4 * t + 60 * np.sin(2 * np.pi * t / 12)
            + np.random.default_rng(seed).normal(0, 12, 72))
    return pd.DataFrame({"month": idx, "sales": vals.round(1)})


def demo_texts():
    return pd.DataFrame({"review": [
        "Absolutely love this product, fantastic quality and great value",
        "Terrible experience, the app kept crashing and support ignored me",
        "Pretty good overall, delivery was fast and packaging was neat",
        "Worst purchase ever, broke within a week, total waste of money",
        "Excellent customer service, they resolved my issue in minutes",
        "Mediocre at best, overpriced for what you actually get",
        "The new update is smooth and the design looks beautiful",
        "Very disappointed, the item did not match the description at all",
        "Solid performance and reliable, would happily recommend it",
        "Awful, slow, and buggy — I regret buying this completely",
        "Great value bundle, my whole family enjoys using it daily",
        "Customer support was rude and unhelpful when I asked for a refund",
    ]})


def metric_row(st, metrics):
    cols = st.columns(len(metrics))
    for col, (name, val) in zip(cols, metrics.items()):
        col.metric(name, "—" if val != val else f"{val:.3f}")


# =========================================================================== #
# TAB 1 — Classification
# =========================================================================== #

def classifier_tab(st, go, px):
    st.subheader("Classification workbench")
    st.caption("Churn · credit risk · loan default · conversion · fraud flags")

    with st.expander("1 · Data", expanded=True):
        use_demo = st.checkbox("Use demo churn dataset", key="clf_demo")
        file = st.file_uploader("Upload a CSV", type=["csv"], key="clf_file")
    df = demo_churn() if use_demo else (load_csv(file, st) if file else None)
    if df is None:
        st.info("Upload a CSV or tick the demo dataset to begin.")
        return

    c = st.columns(4)
    c[0].metric("Rows", f"{df.shape[0]:,}")
    c[1].metric("Columns", f"{df.shape[1]:,}")
    c[2].metric("Missing", f"{int(df.isna().sum().sum()):,}")
    c[3].metric("Duplicates", f"{int(df.duplicated().sum()):,}")
    st.dataframe(df.head(15), use_container_width=True)

    with st.expander("2 · Variables", expanded=True):
        target_opts = suggest_targets(df) or list(df.columns)
        target = st.selectbox("Target (dependent) variable", target_opts, key="clf_target")
        classes = sorted(df[target].dropna().unique().tolist(), key=lambda v: str(v))
        n_classes = len(classes)
        if n_classes < 2:
            st.error("Target needs at least two classes.")
            return
        positive_label = None
        if n_classes == 2:
            positive_label = st.selectbox("Positive class (the '1')", classes,
                                          index=len(classes) - 1, key="clf_pos")
        predictors = st.multiselect(
            "Predictors", [c for c in df.columns if c != target],
            default=[c for c in df.columns if c != target], key="clf_preds")
    if not predictors:
        st.warning("Select at least one predictor.")
        return

    dist = df[target].value_counts(dropna=False).reset_index()
    dist.columns = [target, "count"]
    fig = px.bar(dist, x=target, y="count", color_discrete_sequence=[NAVY])
    fig.update_layout(height=260, margin=dict(t=10, b=10), plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("3 · Model & settings", expanded=True):
        model_choices = [m for m in MODEL_NAMES if m != "XGBoost" or HAS_XGB]
        model_name = st.radio("Method", model_choices, key="clf_model", horizontal=True)
        if not HAS_XGB:
            st.caption("Install `xgboost` to enable XGBoost.")
        s = st.columns(3)
        test_size = s[0].slider("Test size", 0.1, 0.5, 0.25, 0.05, key="clf_test")
        seed = int(s[1].number_input("Seed", 0, 9999, 42, key="clf_seed"))
        cols2 = s[2]
        scale_numeric = cols2.checkbox("Standardise numerics", True, key="clf_scale")
        use_imbalance = cols2.checkbox("Handle imbalance", key="clf_imb")
        do_cv = cols2.checkbox("5-fold CV", key="clf_cv")

        params = {}
        pc = st.columns(3)
        if model_name == "Logistic Regression":
            params["C"] = pc[0].slider("C (inverse reg.)", 0.01, 10.0, 1.0, key="clf_C")
        elif model_name == "Random Forest":
            params["n_estimators"] = pc[0].slider("Trees", 50, 800, 300, 50, key="clf_rf_n")
            params["max_depth"] = _opt_depth(pc[1], "clf_rf_d")
        elif model_name == "Gradient Boosting":
            params["n_estimators"] = pc[0].slider("Rounds", 50, 600, 200, 50, key="clf_gb_n")
            params["learning_rate"] = pc[1].slider("LR", 0.01, 0.5, 0.1, key="clf_gb_lr")
            params["max_depth"] = pc[2].slider("Depth", 1, 8, 3, key="clf_gb_d")
        elif model_name == "Decision Tree":
            params["max_depth"] = _opt_depth(pc[0], "clf_dt_d")
        elif model_name == "XGBoost":
            params["n_estimators"] = pc[0].slider("Rounds", 50, 800, 300, 50, key="clf_xgb_n")
            params["learning_rate"] = pc[1].slider("LR", 0.01, 0.5, 0.1, key="clf_xgb_lr")
            params["max_depth"] = pc[2].slider("Depth", 1, 10, 4, key="clf_xgb_d")
            if use_imbalance and n_classes == 2:
                neg = (df[target] != positive_label).sum()
                pos = max((df[target] == positive_label).sum(), 1)
                params["scale_pos_weight"] = st.slider(
                    "scale_pos_weight", 0.5, max(10.0, neg / pos),
                    float(round(neg / pos, 2)), key="clf_spw")

    if st.button("▶ Run classification", type="primary", key="clf_run"):
        X = df[predictors].copy()
        y, ordered = encode_target(df[target], positive_label=positive_label)
        mask = ~pd.isna(y)
        X, y = X.loc[mask], y[mask].astype(int)
        with st.spinner(f"Training {model_name}…"):
            try:
                res = run_classification(X, y, n_classes, model_name, params,
                                         test_size, scale_numeric, use_imbalance, seed)
            except Exception as e:
                st.error(f"Training failed: {e}")
                return
            cv = (cross_validate_classification(X, y, n_classes, model_name, params,
                                                scale_numeric, use_imbalance, seed)
                  if do_cv else None)
        st.session_state["clf_results"] = {
            **res, "ordered_classes": ordered, "model_name": model_name, "cv": cv}

    if "clf_results" in st.session_state:
        _render_classification(st, go, px, st.session_state["clf_results"])


def _opt_depth(col, key):
    if col.checkbox("Unlimited depth", True, key=key + "_u"):
        return None
    return col.slider("Max depth", 1, 30, 10, key=key + "_s")


def _render_classification(st, go, px, res):
    y_test, y_proba = res["y_test"], res["y_proba"]
    n_classes, ordered = res["n_classes"], res["ordered_classes"]
    st.markdown(f"### Results — {res['model_name']}")

    if n_classes == 2:
        thr = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01, key="clf_thr",
                        help="Lower to catch more positives (higher recall).")
        y_pred = (y_proba[:, 1] >= thr).astype(int)
    else:
        y_pred = np.argmax(y_proba, axis=1)

    metric_row(st, compute_metrics(y_test, y_pred, y_proba, n_classes))

    if res.get("cv"):
        st.markdown("**5-fold cross-validation**")
        st.table(pd.DataFrame([
            {"Metric": k.replace("test_", ""), "Mean": f"{m:.3f}", "Std": f"±{s:.3f}"}
            for k, (m, s) in res["cv"].items()]))

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y_test, y_pred)
        labels = [str(c) for c in ordered]
        fig = px.imshow(cm, text_auto=True, x=[f"Pred: {l}" for l in labels],
                        y=[f"True: {l}" for l in labels],
                        color_continuous_scale=[[0, "white"], [1, NAVY]])
        fig.update_layout(height=360, coloraxis_showscale=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        if n_classes == 2:
            st.markdown("**ROC & precision–recall**")
            pos = y_proba[:, 1]
            fpr, tpr, _ = roc_curve(y_test, pos)
            prec, rec, _ = precision_recall_curve(y_test, pos)
            t1, t2 = st.tabs(["ROC", "PR"])
            with t1:
                f = go.Figure()
                f.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                            line=dict(color=NAVY, width=3),
                            name=f"AUC={auc(fpr, tpr):.3f}"))
                f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                            line=dict(color=GREY, dash="dash"), name="Chance"))
                f.update_layout(height=340, plot_bgcolor="white", margin=dict(t=10, b=10),
                                xaxis_title="FPR", yaxis_title="TPR")
                st.plotly_chart(f, use_container_width=True)
            with t2:
                f = go.Figure()
                f.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                            line=dict(color=BRASS, width=3),
                            name=f"AP={average_precision_score(y_test, pos):.3f}"))
                f.update_layout(height=340, plot_bgcolor="white", margin=dict(t=10, b=10),
                                xaxis_title="Recall", yaxis_title="Precision")
                st.plotly_chart(f, use_container_width=True)
        else:
            st.markdown("**ROC (one-vs-rest)**")
            st.plotly_chart(_multiclass_roc(go, y_test, y_proba, ordered),
                            use_container_width=True)

    imp = res.get("importances")
    if imp is not None and len(imp):
        st.markdown("**Feature importance / coefficients**")
        imp = imp.assign(abs=imp["importance"].abs()).sort_values("abs").tail(20)
        f = px.bar(imp, x="importance", y="feature", orientation="h",
                   color_discrete_sequence=[BRASS])
        f.update_layout(height=max(300, 24 * len(imp)), plot_bgcolor="white",
                        margin=dict(t=10, b=10), yaxis_title="", xaxis_title="")
        st.plotly_chart(f, use_container_width=True)

    with st.expander("Full classification report"):
        rep = classification_report(y_test, y_pred,
                                    target_names=[str(c) for c in ordered],
                                    zero_division=0, output_dict=True)
        st.dataframe(pd.DataFrame(rep).transpose().round(3), use_container_width=True)

    out = res["X_test"].copy()
    out["actual"] = [ordered[i] for i in y_test]
    out["predicted"] = [ordered[i] for i in y_pred]
    if n_classes == 2:
        out["prob_positive"] = y_proba[:, 1]
    buf = io.StringIO(); out.to_csv(buf, index=False)
    st.download_button("⬇ Download predictions (CSV)", buf.getvalue(),
                       "predictions.csv", "text/csv", key="clf_dl")


def _multiclass_roc(go, y_test, y_proba, ordered):
    y_bin = label_binarize(y_test, classes=list(range(len(ordered))))
    palette = [NAVY, BRASS, NAVY_SOFT, GREY, "#7A5C2E", "#4A6B8A"]
    fig = go.Figure()
    for i, cls in enumerate(ordered):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                      line=dict(color=palette[i % len(palette)], width=2.5),
                      name=f"{cls} (AUC={auc(fpr, tpr):.2f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                  line=dict(color="#CCC", dash="dash"), showlegend=False))
    fig.update_layout(height=360, plot_bgcolor="white", margin=dict(t=10, b=10),
                      xaxis_title="FPR", yaxis_title="TPR")
    return fig


# =========================================================================== #
# TAB 2 — Time series
# =========================================================================== #

def timeseries_tab(st, go, px):
    st.subheader("Time-series forecasting")
    st.caption("Exponential Smoothing · ARIMA / SARIMA · LSTM")

    with st.expander("1 · Data", expanded=True):
        use_demo = st.checkbox("Use demo monthly-sales dataset", key="ts_demo")
        file = st.file_uploader("Upload a CSV", type=["csv"], key="ts_file")
    df = demo_timeseries() if use_demo else (load_csv(file, st) if file else None)
    if df is None:
        st.info("Upload a CSV (a date column + a numeric value column) or tick the demo.")
        return
    st.dataframe(df.head(10), use_container_width=True)

    with st.expander("2 · Columns", expanded=True):
        cc = st.columns(3)
        date_col = cc[0].selectbox("Date column", df.columns, key="ts_date")
        num_cols = [c for c in df.columns if c != date_col]
        value_col = cc[1].selectbox("Value column", num_cols or df.columns, key="ts_val")
        freq_key = cc[2].selectbox("Frequency", list(FREQ_MAP.keys()),
                                   index=3, key="ts_freq")

    try:
        series, seasonal_default = prepare_series(df, date_col, value_col, freq_key)
    except Exception as e:
        st.error(f"Could not build a series: {e}")
        return
    if len(series) < 12:
        st.warning("Need at least ~12 observations to forecast reliably.")
        return

    f = go.Figure()
    f.add_trace(go.Scatter(x=list(series.index), y=series.values, mode="lines",
                line=dict(color=NAVY, width=2), name="History"))
    f.update_layout(height=280, plot_bgcolor="white", margin=dict(t=10, b=10),
                    title="Series")
    st.plotly_chart(f, use_container_width=True)

    with st.expander("3 · Method & parameters", expanded=True):
        methods = ["Exponential Smoothing", "ARIMA / SARIMA", "LSTM"]
        method = st.radio("Method", methods, key="ts_method", horizontal=True)
        max_h = max(2, min(len(series) // 2, 60))
        horizon = st.slider("Forecast horizon (steps)", 1, max_h,
                            min(12, max_h), key="ts_h")

        params = {}
        if method == "Exponential Smoothing":
            pc = st.columns(3)
            params["trend"] = pc[0].selectbox("Trend", [None, "add", "mul"], key="ts_es_t")
            params["seasonal"] = pc[1].selectbox("Seasonal", [None, "add", "mul"], key="ts_es_s")
            params["seasonal_periods"] = int(pc[2].number_input(
                "Seasonal periods", 0, 366, seasonal_default or 12, key="ts_es_sp"))
        elif method == "ARIMA / SARIMA":
            auto = st.checkbox("Auto-select (p,d,q) by AIC", key="ts_auto")
            pc = st.columns(3)
            params["p"] = int(pc[0].number_input("p", 0, 5, 1, key="ts_p"))
            params["d"] = int(pc[1].number_input("d", 0, 2, 1, key="ts_d"))
            params["q"] = int(pc[2].number_input("q", 0, 5, 1, key="ts_q"))
            seas = st.checkbox("Seasonal (SARIMA)", key="ts_seas")
            if seas:
                sc = st.columns(4)
                params["P"] = int(sc[0].number_input("P", 0, 3, 1, key="ts_P"))
                params["D"] = int(sc[1].number_input("D", 0, 2, 0, key="ts_D"))
                params["Q"] = int(sc[2].number_input("Q", 0, 3, 1, key="ts_Q"))
                params["m"] = int(sc[3].number_input("m (period)", 2, 366,
                                                     seasonal_default or 12, key="ts_m"))
            params["auto"], params["seasonal_on"] = auto, seas
        else:  # LSTM
            if not HAS_TF:
                st.warning("LSTM needs TensorFlow, which isn't installed. "
                           "Uncomment `tensorflow-cpu` in requirements.txt to enable it. "
                           "The other two methods work now.")
            pc = st.columns(3)
            params["n_lags"] = int(pc[0].number_input("Lags (window)", 2, 60, 12, key="ts_lags"))
            params["units"] = int(pc[1].number_input("LSTM units", 4, 128, 32, key="ts_units"))
            params["epochs"] = int(pc[2].number_input("Epochs", 5, 200, 30, key="ts_ep"))

    if st.button("▶ Run forecast", type="primary", key="ts_run"):
        _run_timeseries(st, go, series, method, params, horizon)


def _run_timeseries(st, go, series, method, params, horizon):
    train, test = series.iloc[:-horizon], series.iloc[-horizon:]
    ci_future = None
    with st.spinner(f"Fitting {method}…"):
        try:
            if method == "Exponential Smoothing":
                back, _ = fit_exp_smoothing(train, horizon, params["trend"],
                                            params["seasonal"], params["seasonal_periods"])
                fut, _ = fit_exp_smoothing(series, horizon, params["trend"],
                                           params["seasonal"], params["seasonal_periods"])
            elif method == "ARIMA / SARIMA":
                if params.get("auto"):
                    order, aic = auto_arima(train)
                    st.info(f"Auto-selected order (p,d,q) = {order}  ·  AIC = {aic:.1f}")
                else:
                    order = (params["p"], params["d"], params["q"])
                sorder = ((params["P"], params["D"], params["Q"], params["m"])
                          if params.get("seasonal_on") else (0, 0, 0, 0))
                back, _ = fit_arima(train, horizon, order, sorder)
                fut, ci_future = fit_arima(series, horizon, order, sorder)
            else:  # LSTM
                if not HAS_TF:
                    st.error("TensorFlow not installed — cannot run LSTM.")
                    return
                back = fit_lstm(train, horizon, params["n_lags"],
                                params["units"], params["epochs"])
                fut = fit_lstm(series, horizon, params["n_lags"],
                               params["units"], params["epochs"])
        except Exception as e:
            st.error(f"Model failed: {e}")
            return

    metric_row(st, forecast_metrics(test.values, np.asarray(back)[:len(test)]))
    st.caption("Metrics are computed on a hold-out of the last "
               f"{horizon} observations (backtest).")

    fidx = future_index(series, horizon)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(series.index), y=series.values, mode="lines",
                  line=dict(color=NAVY, width=2), name="History"))
    fig.add_trace(go.Scatter(x=list(test.index), y=np.asarray(back)[:len(test)],
                  mode="lines", line=dict(color=BRASS, dash="dot", width=2),
                  name="Backtest"))
    fig.add_trace(go.Scatter(x=list(fidx), y=np.asarray(fut), mode="lines+markers",
                  line=dict(color=BRASS, width=3), name="Forecast"))
    if ci_future is not None:
        fig.add_trace(go.Scatter(
            x=list(fidx) + list(fidx)[::-1],
            y=list(ci_future["upper"].values) + list(ci_future["lower"].values)[::-1],
            fill="toself", fillcolor="rgba(194,155,74,0.18)",
            line=dict(color="rgba(0,0,0,0)"), name="95% CI", hoverinfo="skip"))
    fig.update_layout(height=420, plot_bgcolor="white", margin=dict(t=10, b=10),
                      title=f"{method}: history + {horizon}-step forecast")
    st.plotly_chart(fig, use_container_width=True)

    fc_df = pd.DataFrame({"period": list(fidx), "forecast": np.asarray(fut).round(3)})
    buf = io.StringIO(); fc_df.to_csv(buf, index=False)
    st.download_button("⬇ Download forecast (CSV)", buf.getvalue(),
                       "forecast.csv", "text/csv", key="ts_dl")


# =========================================================================== #
# TAB 3 — Text / Web / Social analytics
# =========================================================================== #

def text_tab(st, go, px):
    st.subheader("Text, web & social-media analytics")
    st.caption("Frequency & word cloud · VADER sentiment · LDA topics · TF-IDF keywords")

    with st.expander("1 · Data source", expanded=True):
        source = st.radio("Source", ["CSV column", "Paste text", "Fetch from URL"],
                          key="txt_source", horizontal=True)
        docs = None
        if source == "CSV column":
            use_demo = st.checkbox("Use demo reviews dataset", key="txt_demo")
            file = st.file_uploader("Upload a CSV", type=["csv"], key="txt_file")
            tdf = demo_texts() if use_demo else (load_csv(file, st) if file else None)
            if tdf is not None:
                text_col = st.selectbox("Text column", tdf.columns, key="txt_col")
                docs = tdf[text_col].dropna().astype(str).tolist()
        elif source == "Paste text":
            txt = st.text_area("Paste text — one document per line", height=160,
                               key="txt_paste")
            docs = [ln.strip() for ln in txt.split("\n") if ln.strip()] if txt else None
        else:
            url = st.text_input("URL", key="txt_url",
                                placeholder="https://example.com/article")
            if st.button("Fetch page", key="txt_fetch") and url:
                try:
                    with st.spinner("Fetching…"):
                        st.session_state["txt_fetched"] = fetch_url_text(url)
                    st.success("Fetched.")
                except Exception as e:
                    st.error(f"Could not fetch URL: {e}")
            if st.session_state.get("txt_fetched"):
                docs = split_sentences(st.session_state["txt_fetched"])
                st.caption(f"Extracted {len(docs)} text segments from the page.")

    if not docs:
        st.info("Provide some text to analyse.")
        return
    st.metric("Documents", f"{len(docs):,}")

    with st.expander("2 · Method & parameters", expanded=True):
        method = st.radio(
            "Method", ["Word frequency & cloud", "Sentiment (VADER)",
                       "Topic modelling (LDA)", "Keyword extraction (TF-IDF)"],
            key="txt_method")
        pc = st.columns(3)
        use_sw = pc[0].checkbox("Remove stopwords", True, key="txt_sw")
        extra_raw = pc[1].text_input("Extra stopwords (comma-sep)", key="txt_extra")
        extra = [w for w in extra_raw.split(",")] if extra_raw else None
        top_n = int(pc[2].slider("Top N", 5, 40, 20, key="txt_topn"))
        ngram = (1, 1)
        n_topics = 5
        if method == "Word frequency & cloud":
            ng = st.selectbox("N-gram", ["Unigrams", "Bigrams", "Trigrams"], key="txt_ng")
            ngram = {"Unigrams": (1, 1), "Bigrams": (2, 2), "Trigrams": (3, 3)}[ng]
        if method == "Topic modelling (LDA)":
            n_topics = int(st.slider("Number of topics", 2, 12, 5, key="txt_nt"))

    if st.button("▶ Run analysis", type="primary", key="txt_run"):
        _run_text(st, go, px, docs, method, use_sw, extra, top_n, ngram, n_topics)


def _run_text(st, go, px, docs, method, use_sw, extra, top_n, ngram, n_topics):
    try:
        if method == "Word frequency & cloud":
            freq = word_frequencies(docs, ngram, top_n, use_sw, extra)
            left, right = st.columns([3, 2])
            with left:
                f = px.bar(freq.sort_values("count"), x="count", y="term",
                           orientation="h", color_discrete_sequence=[NAVY])
                f.update_layout(height=max(320, 22 * len(freq)), plot_bgcolor="white",
                                margin=dict(t=10, b=10), yaxis_title="", xaxis_title="")
                st.plotly_chart(f, use_container_width=True)
            with right:
                st.markdown("**Word cloud**")
                try:
                    from wordcloud import WordCloud
                    wc = WordCloud(width=600, height=460, background_color="white",
                                   colormap="cividis").generate_from_frequencies(
                        dict(zip(freq["term"], freq["count"].astype(float))))
                    st.image(wc.to_array(), use_container_width=True)
                except Exception as e:
                    st.caption(f"Word cloud unavailable: {e}")

        elif method == "Sentiment (VADER)":
            sdf = vader_scores(docs)
            pos = (sdf["label"] == "positive").mean()
            neg = (sdf["label"] == "negative").mean()
            metric_row(st, {"Mean compound": sdf["compound"].mean(),
                            "% positive": pos, "% negative": neg})
            left, right = st.columns(2)
            with left:
                f = px.histogram(sdf, x="compound", nbins=25,
                                 color_discrete_sequence=[NAVY])
                f.update_layout(height=320, plot_bgcolor="white", margin=dict(t=10, b=10),
                                title="Distribution of compound sentiment")
                st.plotly_chart(f, use_container_width=True)
            with right:
                counts = sdf["label"].value_counts().reindex(
                    ["negative", "neutral", "positive"]).fillna(0).reset_index()
                counts.columns = ["label", "count"]
                f = px.bar(counts, x="label", y="count", color="label",
                           color_discrete_map={"negative": "#B4453B",
                                               "neutral": GREY, "positive": "#3B7A57"})
                f.update_layout(height=320, plot_bgcolor="white", margin=dict(t=10, b=10),
                                showlegend=False, title="Sentiment classes")
                st.plotly_chart(f, use_container_width=True)
            st.markdown("**Most positive / most negative**")
            show = pd.concat([sdf.nlargest(3, "compound"), sdf.nsmallest(3, "compound")])
            st.dataframe(show[["compound", "label", "text"]].round(3),
                         use_container_width=True)
            buf = io.StringIO(); sdf.to_csv(buf, index=False)
            st.download_button("⬇ Download scored text (CSV)", buf.getvalue(),
                               "sentiment.csv", "text/csv", key="txt_dl_s")

        elif method == "Topic modelling (LDA)":
            if len(docs) < n_topics:
                st.warning(f"Need at least {n_topics} documents for {n_topics} topics.")
                return
            topics = lda_topics(docs, n_topics, 10, use_sw, extra)
            st.markdown("**Discovered topics**")
            st.dataframe(topics, use_container_width=True, hide_index=True)

        else:  # TF-IDF keywords
            kw = tfidf_keywords(docs, top_n, use_sw, extra)
            f = px.bar(kw.sort_values("tfidf"), x="tfidf", y="term",
                       orientation="h", color_discrete_sequence=[BRASS])
            f.update_layout(height=max(320, 22 * len(kw)), plot_bgcolor="white",
                            margin=dict(t=10, b=10), yaxis_title="", xaxis_title="")
            st.plotly_chart(f, use_container_width=True)
    except Exception as e:
        st.error(f"Analysis failed: {e}")


# =========================================================================== #
# Main
# =========================================================================== #

def main():
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px

    st.set_page_config(page_title="Dr Ash Multimodel Advanced BI Studio",
                       page_icon="\U0001F4CA", layout="wide")
    st.markdown(
        f"""
        <style>
        .block-container {{padding-top: 2rem;}}
        h1, h2, h3 {{color: {NAVY};}}
        div[data-testid="stMetricValue"] {{color:{NAVY};}}
        .stTabs [data-baseweb="tab-list"] {{gap: 6px;}}
        .stTabs [aria-selected="true"] {{color:{BRASS};}}
        </style>
        """, unsafe_allow_html=True)

    st.markdown(
        f"<h1 style='margin-bottom:0'>Dr&nbsp;Ash Multimodel Classifier&nbsp;Studio</h1>"
        f"<p style='color:{GREY};margin-top:4px'>Classification · time-series "
        f"forecasting · text, web &amp; social-media analytics — load data, pick a "
        f"method, set parameters, run, and visualise.</p>",
        unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "🎯 Classification", "📈 Time series", "💬 Text / Web / Social"])
    with tab1:
        classifier_tab(st, go, px)
    with tab2:
        timeseries_tab(st, go, px)
    with tab3:
        text_tab(st, go, px)


if __name__ == "__main__":
    main()
