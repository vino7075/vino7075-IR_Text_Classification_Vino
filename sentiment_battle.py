"""
Information Retrieval - CLA-2, Task 2
Text Classification Battle: sentiment of airline customer tweets

    Decision Tree  |  K-Nearest Neighbours  |  SVM  |  Naive Bayes

Vino Bala V  (RA2411026050054)  -  III Year CSE AIML A

Dataset : Twitter US Airline Sentiment (Figure Eight / Crowdflower, Feb 2015)
Run     : python sentiment_battle.py
"""

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)

HERE = Path(__file__).parent
RAW_FILE = HERE / "dataset" / "Tweets.csv"
TABLES = HERE / "tables"
FIGS = HERE / "figures"
TABLES.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

SEED = 11
LABELS = ["negative", "neutral", "positive"]
COLOURS = {"Decision Tree": "#8c564b", "KNN": "#17becf", "SVM": "#9467bd", "Naive Bayes": "#e377c2"}

# Negation words carry the sentiment in phrases like "not happy" or "never again",
# so they are taken OUT of the stop-word list.
NEGATIONS = {"no", "not", "nor", "never", "none", "nothing", "nobody", "nowhere", "cannot", "neither"}
STOP_KEEP_NEGATION = set(ENGLISH_STOP_WORDS) - NEGATIONS
STOP_FULL = set(ENGLISH_STOP_WORDS)

stem = PorterStemmer().stem
console = []


def say(text=""):
    print(text)
    console.append(str(text))


# ------------------------------------------------------------------ data
def load_tweets():
    df = pd.read_csv(RAW_FILE, usecols=["airline_sentiment", "airline", "text"])
    df = df.rename(columns={"airline_sentiment": "sentiment"})
    n_raw = len(df)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    return df, n_raw - len(df)


def tokenize(tweet, stopwords):
    t = tweet.lower()
    t = re.sub(r"https?://\S+", " ", t)          # links
    t = re.sub(r"@\w+", " ", t)                  # @airline / @user mentions
    t = re.sub(r"&amp;|&lt;|&gt;", " ", t)       # html entities
    t = re.sub(r"n't\b", " not", t)              # didn't -> did not
    words = re.findall(r"[a-z]+", t)
    return " ".join(stem(w) for w in words if w not in stopwords and len(w) > 1)


# ------------------------------------------------------------------ models
def get_models():
    return {
        "Decision Tree": DecisionTreeClassifier(min_samples_leaf=2, random_state=SEED),
        "KNN": KNeighborsClassifier(n_neighbors=20, metric="cosine"),
        "SVM": LinearSVC(C=0.5, random_state=SEED),
        "Naive Bayes": MultinomialNB(alpha=0.3),
    }


def vectorizer():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def fit_and_measure(model, Xa, ya, Xb, yb):
    t0 = time.perf_counter()
    model.fit(Xa, ya)
    t_train = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = model.predict(Xb)
    t_test = time.perf_counter() - t0
    p, r, f, _ = precision_recall_fscore_support(yb, pred, average="macro", zero_division=0)
    return pred, dict(Accuracy=accuracy_score(yb, pred), Precision=p, Recall=r, F1=f,
                      TrainAcc=accuracy_score(ya, model.predict(Xa)),
                      TrainTime=t_train, TestTime=t_test)


# ------------------------------------------------------------------ plots
def bar_chart(res):
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for i, row in res.iterrows():
        vals = [row[m] for m in metrics]
        bars = ax.bar(x + (i - 1.5) * 0.2, vals, 0.2, label=row["Model"], color=COLOURS[row["Model"]])
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ax.set_xticks(x, ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"])
    ax.set_ylim(0.4, 0.9)
    ax.set_ylabel("Score")
    ax.set_title("Airline tweet sentiment: four classifiers on the same TF-IDF features")
    ax.legend(ncol=4, loc="upper right", fontsize=8)
    ax.grid(axis="y", ls=":")
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_classifier_comparison.png", dpi=150)
    plt.close(fig)


def confusion_grid(cms):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9))
    for ax, (name, cm) in zip(axes.ravel(), cms.items()):
        ax.imshow(cm, cmap="Purples")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]}\n{100 * cm[i, j] / cm[i].sum():.1f}%", ha="center", va="center",
                        fontsize=10, color="white" if cm[i, j] > cm.max() * 0.55 else "black")
        ax.set_xticks(range(3), LABELS)
        ax.set_yticks(range(3), LABELS)
        ax.set_xlabel("Predicted sentiment")
        ax.set_ylabel("Actual sentiment")
        ax.set_title(name)
    fig.suptitle("Confusion matrices (count and row %)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_confusion_matrices.png", dpi=150)
    plt.close(fig)


def recall_chart(per_class):
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    per_class.plot(kind="bar", ax=ax, color=[COLOURS[c] for c in per_class.columns], width=0.8)
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.05)
    ax.set_title("Recall per sentiment class (minority classes are harder)")
    ax.set_xticklabels(per_class.index, rotation=0)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", ls=":")
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_recall_per_class.png", dpi=150)
    plt.close(fig)


def whatif_chart(w):
    piv = w.pivot(index="Model", columns="Stop-word list", values="F1").loc[list(COLOURS)]
    piv = piv[["Standard list (negations removed)", "Negations kept (main setup)"]]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    piv.plot(kind="bar", ax=ax, color=["#bdbdbd", "#2b8cbe"], width=0.7)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", fontsize=8, padding=2)
    ax.set_ylim(0.4, 0.85)
    ax.set_ylabel("Macro F1")
    ax.set_xlabel("")
    ax.set_xticklabels(piv.index, rotation=0)
    ax.set_title("What-if: removing vs keeping negation words")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_whatif_negation.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    df, dups = load_tweets()
    say("SENTIMENT CLASSIFICATION BATTLE - Twitter US Airline Sentiment")
    say("=" * 62)
    say(f"Tweets after removing {dups} duplicates : {len(df)}")
    dist = df["sentiment"].value_counts().reindex(LABELS)
    for lab in LABELS:
        say(f"   {lab:<9}: {dist[lab]:>5}  ({100 * dist[lab] / len(df):.1f}%)")
    say(f"Majority-class baseline accuracy   : {dist.max() / len(df):.4f}")
    dist.to_csv(TABLES / "class_distribution.csv", header=["tweets"])

    df["tokens"] = df["text"].apply(lambda t: tokenize(t, STOP_KEEP_NEGATION))
    empty = int((df["tokens"].str.len() == 0).sum())
    df = df[df["tokens"].str.len() > 0].reset_index(drop=True)
    say(f"Tweets left empty after preprocessing (dropped): {empty}  ->  {len(df)} tweets used")
    say(f"Avg words per tweet : {df['text'].str.split().str.len().mean():.1f} -> "
        f"{df['tokens'].str.split().str.len().mean():.1f} tokens after preprocessing")
    with open(TABLES / "preprocessing_examples.txt", "w") as fh:
        for i in [1, 3, 14]:
            fh.write(f"[{df.sentiment[i]}]\n  tweet : {df.text[i]}\n  tokens: {df.tokens[i]}\n\n")

    idx_tr, idx_te = train_test_split(df.index, test_size=0.20, stratify=df["sentiment"], random_state=SEED)
    y_tr, y_te = df.loc[idx_tr, "sentiment"], df.loc[idx_te, "sentiment"]
    vec = vectorizer()
    X_tr = vec.fit_transform(df.loc[idx_tr, "tokens"])
    X_te = vec.transform(df.loc[idx_te, "tokens"])
    say(f"Train tweets: {len(idx_tr)} | Test tweets: {len(idx_te)} | stratified 80/20")
    say(f"TF-IDF features (unigrams + bigrams): {X_tr.shape[1]}")

    rows, cms, preds = [], {}, {}
    for name, model in get_models().items():
        pred, m = fit_and_measure(model, X_tr, y_tr, X_te, y_te)
        rows.append({"Model": name, **m})
        preds[name] = pred
        cms[name] = confusion_matrix(y_te, pred, labels=LABELS)
        say(f"\n>> {name}")
        say(classification_report(y_te, pred, labels=LABELS, digits=4, zero_division=0).rstrip())

    res = pd.DataFrame(rows)
    res.to_csv(TABLES / "results.csv", index=False)
    say("\nCOMPARISON TABLE (test set, macro averages)")
    say(res.round(4).to_string(index=False))

    recall = pd.DataFrame({n: precision_recall_fscore_support(y_te, p, labels=LABELS, zero_division=0)[1]
                           for n, p in preds.items()}, index=LABELS)
    recall.to_csv(TABLES / "recall_per_class.csv")
    say("\nRecall per class")
    say(recall.round(4).to_string())

    best = res.loc[res["F1"].idxmax(), "Model"]
    say(f"\nBest model (macro F1): {best}")
    cm = cms[best]
    say(pd.DataFrame(cm, index=[f"true {l}" for l in LABELS], columns=[f"pred {l}" for l in LABELS]).to_string())
    for name, c in cms.items():
        pd.DataFrame(c, index=LABELS, columns=LABELS).to_csv(TABLES / f"confusion_{name.replace(' ', '_')}.csv")

    test = df.loc[idx_te, ["sentiment", "text"]].assign(predicted=preds[best])
    test[test.sentiment != test.predicted].to_csv(TABLES / f"misclassified_{best.replace(' ', '_')}.csv",
                                                  index=False)

    feats = np.array(vec.get_feature_names_out())
    svm = LinearSVC(C=0.5, random_state=SEED).fit(X_tr, y_tr)
    with open(TABLES / "svm_top_terms.txt", "w") as fh:
        for i, c in enumerate(svm.classes_):
            fh.write(f"{c:<9}: {', '.join(feats[np.argsort(-svm.coef_[i])[:15]])}\n")

    bar_chart(res)
    confusion_grid(cms)
    recall_chart(recall)

    # ---------------- WHAT-IF: what happens if negation words are removed as stop words?
    say("\nWHAT-IF: standard stop-word list (negations removed) vs keeping negation words")
    wi = []
    for label, sw in [("Standard list (negations removed)", STOP_FULL),
                      ("Negations kept (main setup)", STOP_KEEP_NEGATION)]:
        toks = df["text"].apply(lambda t: tokenize(t, sw))
        v = vectorizer()
        A, B = v.fit_transform(toks.loc[idx_tr]), v.transform(toks.loc[idx_te])
        for name, model in get_models().items():
            pred, m = fit_and_measure(model, A, y_tr, B, y_te)
            neg_recall = precision_recall_fscore_support(y_te, pred, labels=LABELS, zero_division=0)[1][0]
            wi.append({"Stop-word list": label, "Model": name, "Accuracy": m["Accuracy"], "F1": m["F1"],
                       "Negative recall": neg_recall})
    wi = pd.DataFrame(wi)
    wi.to_csv(TABLES / "whatif_negation.csv", index=False)
    say(wi.pivot(index="Model", columns="Stop-word list", values="F1").round(4).to_string())
    whatif_chart(wi)

    (TABLES / "console_log.txt").write_text("\n".join(console) + "\n")
    say("\nSaved tables/ and figures/")


if __name__ == "__main__":
    main()
