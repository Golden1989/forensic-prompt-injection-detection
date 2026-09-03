"""
Objetivo: medir performance, priorizando recall da classe anômala.
"""
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Nesse ponto você precisa de um conjunto de teste com exemplos MALICIOSOS rotulados
# (vem da etapa 5 — dados sintéticos + qualquer exemplo real de dataset público de injection)

# predict() retorna 1 (normal) ou -1 (anomalia) -> converter para 0/1
y_pred_iso = iso_forest.predict(X_test)
y_pred_iso = np.where(y_pred_iso == -1, 1, 0)  # 1 = anômalo/malicioso

print(confusion_matrix(y_test, y_pred_iso))
print(classification_report(y_test, y_pred_iso, target_names=["normal", "malicioso"]))