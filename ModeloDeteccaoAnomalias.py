"""
Objetivo: treinar o detector só com dados normais (essa é a lógica de detecção de anomalia — o modelo aprende "o que é normal" e sinaliza desvios).
"""
import os

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
import numpy as np
import joblib 

embeddings = np.load("data/train_embeddings.npy")
X_train, X_val = train_test_split(embeddings, test_size=0.2, random_state=42)

# OneClassSVM com kernel RBF escala mal com muitas amostras (custo entre
# quadratico e cubico), entao treina numa amostra reduzida de X_train.
rng = np.random.RandomState(42)
sample_idx = rng.choice(X_train.shape[0], size=5000, replace=False)
X_train_ocsvm = X_train[sample_idx]

# Modelo A
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.05,  # % esperado de anomalias -> ajustar depois via validação
    random_state=42,
)
iso_forest.fit(X_train)

# Modelo B
ocsvm = OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=0.05,  # equivalente ao "contamination" do IsolationForest
)
ocsvm.fit(X_train_ocsvm)

os.makedirs("data", exist_ok=True)
np.save("data/val_embeddings.npy", X_val)
joblib.dump(iso_forest, "data/modelo_isolation_forest.joblib")
joblib.dump(ocsvm, "data/modelo_ocsvm.joblib")