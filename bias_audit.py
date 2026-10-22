"""
Sentiment Classifier Bias Audit — synthetic teaching demo.
Run: python bias_audit.py
Dependencies: pandas, numpy, scikit-learn, matplotlib
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

RANDOM_STATE = 42
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
rng = np.random.default_rng(RANDOM_STATE)
pos = ["I absolutely love this product", "This is wonderful and excellent", "Really happy with the service", "Great quality, I recommend it", "Fantastic experience, very good", "I am pleased with this purchase", "Brilliant result and lovely design", "This works well and feels amazing"]
neg = ["I really hate this product", "This is awful and disappointing", "Very unhappy with the service", "Poor quality, do not recommend", "Terrible experience, very bad", "I regret this purchase", "Horrible result and frustrating design", "This does not work well"]
neutral = ["The product arrived yesterday", "I used the service today", "The package contains two items", "This is the current version", "The order was processed in the morning", "The item has a standard design"]
genders = ["Woman", "Man", "Non-binary"]
styles = ["Standard-style", "Colloquial-style"]
rows = []
for _ in range(900):
    label = int(rng.random() < .5)
    text = rng.choice(neutral) if rng.random() < .12 else rng.choice(pos if label else neg)
    if text in neutral: label = int(rng.random() < .5)
    gender = rng.choice(genders, p=[.46,.46,.08])
    style = rng.choice(styles, p=[.65,.35])
    if style == "Colloquial-style":
        text = text.replace("wonderful","so good").replace("excellent","awesome").replace("really","so")
        text = text.replace("awful","so bad").replace("terrible","really bad")
        text += rng.choice(["!!", " :)", " lol", ""])
    if gender == "Non-binary" and rng.random() < .25: text = "Honestly, " + text.lower()
    rows.append({"text":text,"sentiment":label,"gender_group":gender,"style_group":style})
df = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)

X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
    df.text, df.sentiment, df[["gender_group","style_group"]],
    test_size=.25, random_state=RANDOM_STATE, stratify=df.sentiment)
model = Pipeline([("tfidf",TfidfVectorizer(ngram_range=(1,2),min_df=2)),
                  ("classifier",LogisticRegression(max_iter=1000,random_state=RANDOM_STATE))])
model.fit(X_train,y_train); pred = model.predict(X_test)

def metrics(y, p):
    tn,fp,fn,tp = confusion_matrix(y,p,labels=[0,1]).ravel()
    return {"n":len(y),"accuracy":accuracy_score(y,p),
            "precision_positive":precision_score(y,p,zero_division=0),
            "recall_positive":recall_score(y,p,zero_division=0),
            "f1_positive":f1_score(y,p,zero_division=0),
            "false_positive_rate":fp/(fp+tn) if fp+tn else np.nan,
            "false_negative_rate":fn/(fn+tp) if fn+tp else np.nan,
            "positive_prediction_rate":float(np.mean(p))}
results=meta_test.reset_index(drop=True).copy()
results["y_true"]=np.asarray(y_test); results["y_pred"]=pred
for col in ["gender_group","style_group"]:
    records=[]
    for group, part in results.groupby(col):
        row=metrics(part.y_true,part.y_pred); row[col]=group; records.append(row)
    audit=pd.DataFrame(records).set_index(col)
    audit.to_csv(os.path.join(OUTPUT_DIR,f"{col}_metrics.csv"))
    for metric in ["accuracy","false_positive_rate","false_negative_rate"]:
        ax=audit[metric].plot(kind="bar",ylim=(0,1),figsize=(7,4),title=f"{metric} by {col}")
        ax.set_ylabel(metric); plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR,f"{metric}_by_{col}.png"),dpi=160); plt.close()
overall=pd.DataFrame([metrics(y_test,pred)])
overall.to_csv(os.path.join(OUTPUT_DIR,"overall_metrics.csv"),index=False)
print("Overall metrics:\n",overall.round(3).to_string(index=False))
print("Outputs saved in",OUTPUT_DIR)
