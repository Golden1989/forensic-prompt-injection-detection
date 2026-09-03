"""
Objetivo: treinar o detector só com dados normais (essa é a lógica de detecção de anomalia — o modelo aprende "o que é normal" e sinaliza desvios).
"""
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split

X_train, X_val = train_test_split(embeddings, test_size=0.2, random_state=42)

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
ocsvm.fit(X_train)
