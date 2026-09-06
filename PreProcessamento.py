"""
Objetivo: limpar texto e gerar embeddings (representação numérica).
"""
# pip install sentence-transformers scikit-learn

import re
import pandas as pd
import os
import numpy as np
from sentence_transformers import SentenceTransformer

df_normal = pd.read_csv("data/prompts_normais.csv")

def limpar_texto(texto: str) -> str:
    texto = texto.strip()
    texto = re.sub(r"\s+", " ", texto)  # espaços múltiplos
    return texto

df_normal["text_clean"] = df_normal["text"].astype(str).apply(limpar_texto)

# Embeddings: usar um modelo pré-treinado leve, roda bem em CPU
modelo_embedding = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = modelo_embedding.encode(
    df_normal["text_clean"].tolist(),
    show_progress_bar=True,
    batch_size=32,
)
# embeddings.shape = (n_amostras, 384) -> essa é sua "matriz de features"

os.makedirs("data", exist_ok=True)
np.save("data/train_embeddings.npy", embeddings)
np.save("data/train_labels.npy", df_normal["label"].to_numpy())