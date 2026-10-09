"""
Objetivo: comparar combinações de modelo de embedding + detector de anomalia
usando SÓ o conjunto de calibração, para escolher a melhor antes do teste.

Motivação: a avaliação baseline (Avaliacao.py, MiniLM + OneClassSVM) teve
ROC-AUC 0,72 no teste e, para bloquear 95% dos ataques, bloqueava 88% dos
prompts normais. Um diagnóstico na calibração mostrou que um detector por
distância aos vizinhos mais próximos (kNN) já separava bem melhor.

Todos os detectores continuam treinados só com prompts normais (detecção de
anomalia). O conjunto de teste não é carregado neste script.

Os embeddings de cada modelo ficam em cache em data/embeddings/<modelo>/.

Uso:
    python experimentos_detector.py
"""
import os
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM

from aumentodedados import limpar_texto

MODELOS_EMBEDDING = [
    "all-MiniLM-L6-v2",  # baseline atual, 384 dimensões
    "BAAI/bge-small-en-v1.5",  # 384 dimensões
    "all-mpnet-base-v2",  # 768 dimensões
    "BAAI/bge-base-en-v1.5",  # 768 dimensões
]

VALORES_K = [1, 5, 10, 20]
# Embeddings normalizados: ||x - y||² = 2 - 2·cos(x, y). gamma="scale" é o
# padrão usado no baseline; os outros valores deixam o kernel mais "local".
GRADE_OCSVM = [{"gamma": g, "nu": nu} for g in ["scale", 1.0, 2.0, 4.0] for nu in [0.05, 0.1]]
AMOSTRA_OCSVM = 5000  # mesmo tamanho do baseline (custo do kernel RBF)

PASTA_CACHE = "data/embeddings"
ARQUIVO_SAIDA = "resultados/experimentos_calibracao.csv"


def pasta_modelo(nome: str) -> str:
    return os.path.join(PASTA_CACHE, nome.replace("/", "__"))


def embeddings(nome: str) -> tuple[np.ndarray, np.ndarray]:
    """Embeddings (normalizados) de todos os prompts normais e maliciosos,
    com a mesma limpeza do PreProcessamento.py. Calcula uma vez e guarda."""
    pasta = pasta_modelo(nome)
    caminho_n, caminho_m = os.path.join(pasta, "normais.npy"), os.path.join(pasta, "maliciosos.npy")
    if os.path.exists(caminho_n) and os.path.exists(caminho_m):
        return np.load(caminho_n), np.load(caminho_m)

    os.makedirs(pasta, exist_ok=True)
    modelo = SentenceTransformer(nome)
    normais = pd.read_csv("data/prompts_normais.csv")["text"].astype(str).map(limpar_texto).tolist()
    maliciosos = pd.read_csv("data/prompts_maliciosos.csv")["text"].astype(str).map(limpar_texto).tolist()
    inicio = time.time()
    emb_n = modelo.encode(normais, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    emb_m = modelo.encode(maliciosos, batch_size=32, normalize_embeddings=True)
    print(f"  embeddings de {nome} em {time.time() - inicio:.0f}s")
    np.save(caminho_n, emb_n)
    np.save(caminho_m, emb_m)
    return emb_n, emb_m


def montar_treino_e_calibracao(emb_n: np.ndarray, emb_m: np.ndarray):
    """Refaz exatamente as divisões do pipeline (mesmos random_state), para
    que treino e calibração tenham os MESMOS prompts em qualquer modelo de
    embedding:
      - ModeloDeteccaoAnomalias.py: normais 80% treino / 20% validação
      - aumentodedados.py: validação 60% calibração / 40% teste;
        maliciosos 60/40 estratificado por técnica
    Devolve só treino e calibração; a parte de teste é descartada aqui.
    """
    idx_treino, idx_val = train_test_split(np.arange(len(emb_n)), test_size=0.2, random_state=42)
    idx_val_calib, _ = train_test_split(np.arange(len(idx_val)), test_size=0.4, random_state=42)

    tecnicas = pd.read_csv("data/prompts_maliciosos.csv")["tecnica"]
    idx_mal_calib, _ = train_test_split(np.arange(len(emb_m)), test_size=0.4, random_state=42, stratify=tecnicas)

    X_treino = emb_n[idx_treino]
    X_calib = np.vstack([emb_n[idx_val][idx_val_calib], emb_m[idx_mal_calib]])
    y_calib = np.concatenate([np.zeros(len(idx_val_calib), dtype=int), np.ones(len(idx_mal_calib), dtype=int)])
    return X_treino, X_calib, y_calib


def taxa_falso_positivo_no_recall(y: np.ndarray, scores: np.ndarray, recall_alvo: float) -> float:
    """Fração dos prompts normais bloqueados pelo limiar que pega pelo menos
    recall_alvo dos ataques (o mesmo critério de Avaliacao.py)."""
    precisao, recall, limiares = precision_recall_curve(y, scores)
    validos = np.where(recall[:-1] >= recall_alvo)[0]
    limiar = limiares[validos[np.argmax(precisao[validos])]]
    return float(((scores >= limiar) & (y == 0)).sum() / (y == 0).sum())


def resumo(y: np.ndarray, scores: np.ndarray) -> dict:
    return {
        "roc_auc": roc_auc_score(y, scores),
        "pr_auc": average_precision_score(y, scores),
        "fp_no_recall_90": taxa_falso_positivo_no_recall(y, scores, 0.90),
        "fp_no_recall_95": taxa_falso_positivo_no_recall(y, scores, 0.95),
    }


def detectores(X_treino: np.ndarray, X_calib: np.ndarray):
    """Gera (nome, configuração, score de anomalia na calibração) para cada
    detector. Em todos, score maior = mais suspeito."""
    floresta = IsolationForest(n_estimators=100, random_state=42).fit(X_treino)
    yield "IsolationForest", "n_estimators=100", -floresta.decision_function(X_calib)

    rng = np.random.RandomState(42)
    amostra = X_treino[rng.choice(len(X_treino), size=AMOSTRA_OCSVM, replace=False)]
    for params in GRADE_OCSVM:
        svm = OneClassSVM(kernel="rbf", **params).fit(amostra)
        yield "OneClassSVM", f"gamma={params['gamma']}, nu={params['nu']}", -svm.decision_function(X_calib)

    # Distância de cosseno média aos k vizinhos mais próximos entre os
    # prompts normais de treino: quanto mais longe de tudo que é normal,
    # mais suspeito.
    vizinhos = NearestNeighbors(n_neighbors=max(VALORES_K), metric="cosine", algorithm="brute").fit(X_treino)
    distancias, _ = vizinhos.kneighbors(X_calib)
    for k in VALORES_K:
        yield "kNN", f"k={k}", distancias[:, :k].mean(axis=1)


if __name__ == "__main__":
    os.makedirs("resultados", exist_ok=True)
    linhas = []
    for nome in MODELOS_EMBEDDING:
        print(f"\n### {nome}")
        emb_n, emb_m = embeddings(nome)
        X_treino, X_calib, y_calib = montar_treino_e_calibracao(emb_n, emb_m)

        if nome == "all-MiniLM-L6-v2":
            # Confere que as divisões refeitas aqui batem com as do pipeline.
            assert np.allclose(X_calib, np.load("data/calibracao_embeddings.npy"), atol=1e-5), \
                "a calibração refeita não bate com data/calibracao_embeddings.npy"

        for detector, config, scores in detectores(X_treino, X_calib):
            linha = {"embedding": nome, "detector": detector, "config": config, **resumo(y_calib, scores)}
            linhas.append(linha)
            print(
                f"  {detector:16s} {config:24s} ROC-AUC {linha['roc_auc']:.3f}  PR-AUC {linha['pr_auc']:.3f}  "
                f"FP@90% {linha['fp_no_recall_90']:.1%}  FP@95% {linha['fp_no_recall_95']:.1%}"
            )

    tabela = pd.DataFrame(linhas).sort_values("pr_auc", ascending=False)
    tabela.to_csv(ARQUIVO_SAIDA, index=False)
    print("\n=== Top 10 por PR-AUC (calibração) ===")
    print(tabela.head(10).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nTabela completa em {ARQUIVO_SAIDA}")
