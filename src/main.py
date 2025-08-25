import os
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Semente para reprodutibilidade
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Colunas principais
ID_COL = "id"
TEXT_COL = "text"
LABEL_COL = "label"

# Diretórios
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================
# 1. Carregar dados
# ============================
print("📥 Lendo dados...")
train = pd.read_json(os.path.join(INPUT_DIR, "train.jsonl"), lines=True)
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
sample_sub = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))

print("\nTrain:")
print(train.head())
print("\nTest:")
print(test.head())

# ============================
# 2. Pré-processamento básico
# ============================
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@[\w_]+")
MULTISPACE_RE = re.compile(r"\s{2,}")

def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text
    t = URL_RE.sub(" <URL> ", t)
    t = MENTION_RE.sub(" <USER> ", t)
    t = MULTISPACE_RE.sub(" ", t)
    return t.strip()

train["_text"] = train[TEXT_COL].astype(str).map(basic_clean)
test["_text"] = test[TEXT_COL].astype(str).map(basic_clean)

# ============================
# 3. Baseline: TF-IDF + Modelos Lineares
# ============================
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from scipy.sparse import hstack

# Configuração de validação cruzada
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# Vetorizadores
word_vectorizer = TfidfVectorizer(
    analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.95,
    lowercase=False, sublinear_tf=True
)
char_vectorizer = TfidfVectorizer(
    analyzer="char", ngram_range=(3, 5), min_df=2, max_df=0.95,
    lowercase=False, sublinear_tf=True
)

print("\n🔄 Ajustando TF-IDF...")
Xw = word_vectorizer.fit_transform(train["_text"])
Xc = char_vectorizer.fit_transform(train["_text"])
X = hstack([Xw, Xc])

Xt_w = word_vectorizer.transform(test["_text"])
Xt_c = char_vectorizer.transform(test["_text"])
Xt = hstack([Xt_w, Xt_c])

y = train[LABEL_COL].values

# Modelos para comparação
models = {
    "logreg": LogisticRegression(max_iter=2000, n_jobs=4, C=2.0, class_weight="balanced"),
    "linsvm": LinearSVC(C=1.0)
}

cv_scores = {name: [] for name in models}

print("\n=== Validação Cruzada (Balanced Accuracy) ===")
for name, model in models.items():
    scores = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        clf = model
        clf.fit(X[tr_idx], y[tr_idx])
        if name == "linsvm":
            preds = clf.decision_function(X[va_idx])
            yhat = (preds >= 0).astype(int)
        else:
            yhat = clf.predict(X[va_idx])
        score = balanced_accuracy_score(y[va_idx], yhat)
        scores.append(score)
        print(f"{name} | fold {fold}: {score:.4f}")
    print(f"{name} | média: {np.mean(scores):.4f}")
    cv_scores[name] = scores

# Selecionar o melhor modelo
best_name = max(cv_scores, key=lambda k: np.mean(cv_scores[k]))
print(f"\n🏆 Melhor modelo: {best_name}")

final_model = models[best_name]
final_model.fit(X, y)

# Previsão final
if best_name == "linsvm":
    preds_test = final_model.decision_function(Xt)
    test_labels = (preds_test >= 0).astype(int)
else:
    test_labels = final_model.predict(Xt)

# ============================
# 4. Salvar submissão
# ============================
submission = test[[ID_COL]].copy()
submission[LABEL_COL.upper()] = test_labels

sub_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(sub_path, index=False)

print(f"\n✅ Submissão salva com sucesso: {sub_path}")
