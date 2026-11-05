"""
TESTE FINAL — roda UMA ÚNICA VEZ, no conjunto de teste, para ver se o
detector por trechos generaliza além da calibração. Avaliado antes numa
cópia isolada do projeto (fora do git), aprovado pelo orientador, e agora
reproduzido aqui no repositório oficial para registro.

Detector avaliado: kNN (distância de cosseno), embeddings all-MiniLM-L6-v2,
k=5, variante "combinado" = max(score do prompt inteiro, score do trecho
mais suspeito), trechos com tamanho mínimo de 20 caracteres. Essa foi a
configuração escolhida em experimentos_detector.py e experimentos_trechos.py,
usando só a calibração.

Tudo que podia ser ajustado (modelo de embedding, k, variante, tamanho
mínimo, limiar de decisão) já foi decidido ANTES e usando só a calibração.
O limiar é escolhido aqui mesmo, mas só com a calibração (RECALL_ALVO=0.95,
igual ao Avaliacao.py) -- e só depois aplicado ao teste, sem reajuste.

Uso:
    python teste_final_segmentos.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from Avaliacao import (
    RECALL_ALVO,
    escolher_limiar,
    imprimir,
    metadados_maliciosos,
    metricas,
    recall_por_grupo,
)
from experimentos_detector import embeddings, indices_divisao
from experimentos_trechos import embeddings_trechos, max_por_prompt

NOME_EMBEDDING = "all-MiniLM-L6-v2"
TAMANHO_MINIMO = 20
K = 5

PASTA_RESULTADOS = "resultados"
ARQUIVO_SAIDA = f"{PASTA_RESULTADOS}/teste_final_segmentos.json"


def indices_normais_teste() -> np.ndarray:
    """Posições, em prompts_normais.csv, dos prompts normais do teste final.
    Reproduz exatamente os mesmos splits (random_state=42) usados em
    indices_divisao() e em aumentodedados.montar_conjuntos(), mas capturando
    o lado que eles descartam (o teste)."""
    n_normais = len(pd.read_csv("data/prompts_normais.csv"))
    _, idx_val = train_test_split(np.arange(n_normais), test_size=0.2, random_state=42)
    _, idx_val_teste = train_test_split(np.arange(len(idx_val)), test_size=0.4, random_state=42)
    return idx_val[idx_val_teste]


def salvar_graficos_finais(scores_teste: np.ndarray, y_teste: np.ndarray, limiar: float) -> None:
    """Curva precisão-recall e distribuição dos scores do detector final no
    teste. Nomes de arquivo diferentes dos do baseline (curva_pr_teste.png),
    para manter os dois registrados lado a lado."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    p, r, _ = precision_recall_curve(y_teste, scores_teste)
    ax.plot(r, p, label=f"kNN por trechos (PR-AUC {average_precision_score(y_teste, scores_teste):.3f})")
    ax.axhline(y_teste.mean(), color="grey", linestyle=":", label=f"aleatório ({y_teste.mean():.3f})")
    ax.set_xlabel("Recall (ataques bloqueados)")
    ax.set_ylabel("Precisão (bloqueios corretos)")
    ax.set_title("Curva precisão-recall — detector final, conjunto de teste")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{PASTA_RESULTADOS}/teste_final_curva_pr.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bins = np.linspace(scores_teste.min(), scores_teste.max(), 60)
    ax.hist(scores_teste[y_teste == 0], bins=bins, alpha=0.6, density=True, label="normal")
    ax.hist(scores_teste[y_teste == 1], bins=bins, alpha=0.6, density=True, label="malicioso")
    ax.axvline(limiar, color="black", linestyle="--", label="limiar")
    ax.set_xlabel("Score de anomalia (maior = mais suspeito)")
    ax.set_ylabel("Densidade")
    ax.set_title("Distribuição dos scores — detector final, teste")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{PASTA_RESULTADOS}/teste_final_distribuicao_scores.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)
    idx = indices_divisao()
    normais = pd.read_csv("data/prompts_normais.csv")["text"].astype(str)
    maliciosos = pd.read_csv("data/prompts_maliciosos.csv")

    # ---------- textos, na MESMA ordem dos .npy já congelados ----------
    textos_treino = normais.iloc[idx["treino"]].tolist()
    textos_calib = normais.iloc[idx["normais_calib"]].tolist() + maliciosos["text"].iloc[idx["maliciosos_calib"]].tolist()
    y_calib = np.load("data/calibracao_labels.npy").astype(int)

    meta_teste = metadados_maliciosos("teste")  # já confere contra test_embeddings.npy
    idx_normais_teste = indices_normais_teste()
    textos_normais_teste = normais.iloc[idx_normais_teste].tolist()
    textos_teste = textos_normais_teste + meta_teste["text"].tolist()
    y_teste = np.load("data/test_labels.npy").astype(int)
    assert len(textos_normais_teste) == int((y_teste == 0).sum()), "normais do teste não batem com test_labels.npy"

    # ---------- embeddings do prompt inteiro (já congelados) ----------
    emb_n, _ = embeddings(NOME_EMBEDDING)
    X_treino = emb_n[idx["treino"]]
    X_calib = np.load("data/calibracao_embeddings.npy")
    X_teste = np.load("data/test_embeddings.npy")

    viz_inteiro = NearestNeighbors(n_neighbors=K, metric="cosine", algorithm="brute").fit(X_treino)
    s_inteiro_calib = viz_inteiro.kneighbors(X_calib)[0].mean(axis=1)
    s_inteiro_teste = viz_inteiro.kneighbors(X_teste)[0].mean(axis=1)

    # ---------- embeddings por trechos (referência e calibração já em cache; teste é novo) ----------
    emb_ref, _ = embeddings_trechos(NOME_EMBEDDING, textos_treino, TAMANHO_MINIMO, "treino")
    emb_cal, dono_cal = embeddings_trechos(NOME_EMBEDDING, textos_calib, TAMANHO_MINIMO, "calibracao")
    print("Calculando embeddings dos trechos do TESTE (primeira vez, sem cache)...")
    emb_tes, dono_tes = embeddings_trechos(NOME_EMBEDDING, textos_teste, TAMANHO_MINIMO, "teste")

    viz_trechos = NearestNeighbors(n_neighbors=K, metric="cosine", algorithm="brute").fit(emb_ref)
    s_trechos_calib = max_por_prompt(viz_trechos.kneighbors(emb_cal)[0].mean(axis=1), dono_cal, len(textos_calib))
    s_trechos_teste = max_por_prompt(viz_trechos.kneighbors(emb_tes)[0].mean(axis=1), dono_tes, len(textos_teste))

    s_combinado_calib = np.maximum(s_inteiro_calib, s_trechos_calib)
    s_combinado_teste = np.maximum(s_inteiro_teste, s_trechos_teste)

    # ---------- limiar: escolhido SÓ com a calibração, antes de olhar o teste ----------
    limiar = escolher_limiar(y_calib, s_combinado_calib, RECALL_ALVO)
    print(f"\n=== CALIBRAÇÃO — kNN por trechos (k={K}, min={TAMANHO_MINIMO}, combinado) ===")
    imprimir("", metricas(y_calib, s_combinado_calib, limiar))
    print(f"  limiar escolhido (recall >= {RECALL_ALVO:.0%} na calibração) = {limiar:.4f}")

    # ---------- TESTE, uma única vez, limiar já congelado ----------
    print("\n=== TESTE FINAL — detector e limiar congelados da calibração ===")
    m_teste = metricas(y_teste, s_combinado_teste, limiar)
    imprimir("kNN por trechos (final)", m_teste)

    s_mal_teste = s_combinado_teste[y_teste == 1]
    por_tecnica = recall_por_grupo(s_mal_teste, limiar, meta_teste["tecnica"])
    print(f"\n=== TESTE FINAL — recall por técnica ===")
    print(por_tecnica.to_string(float_format=lambda v: f"{v:.3f}"))

    com_marcador = meta_teste["tecnica"].isin(["hidden instruction", "indirect injection"]).values
    por_marcador = recall_por_grupo(
        s_mal_teste[com_marcador],
        limiar,
        meta_teste.loc[com_marcador, "marcador_ok"].map({True: "com marcador", False: "naturalizado"}),
    )
    print(f"\n=== TESTE FINAL — hidden + indirect: óbvio vs. naturalizado ===")
    print(por_marcador.to_string(float_format=lambda v: f"{v:.3f}"))

    # ---------- comparação com o baseline antigo, pra contexto ----------
    with open(f"{PASTA_RESULTADOS}/avaliacao_sintetica.json", encoding="utf-8") as f:
        baseline = json.load(f)
    baseline_escolhido = baseline["modelo_escolhido"]
    m_baseline = baseline["teste"][baseline_escolhido]
    print(f"\n=== COMPARAÇÃO — baseline ({baseline_escolhido}) vs. detector novo ===")
    print(f"  baseline:      recall {m_baseline['recall']:.3f}  falso positivo {m_baseline['taxa_falso_positivo']:.3%}  ROC-AUC {m_baseline['roc_auc']:.3f}")
    print(f"  kNN trechos:   recall {m_teste['recall']:.3f}  falso positivo {m_teste['taxa_falso_positivo']:.3%}  ROC-AUC {m_teste['roc_auc']:.3f}")

    salvar_graficos_finais(s_combinado_teste, y_teste, limiar)

    with open(ARQUIVO_SAIDA, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {"embedding": NOME_EMBEDDING, "k": K, "tamanho_minimo": TAMANHO_MINIMO, "variante": "combinado"},
                "recall_alvo": RECALL_ALVO,
                "limiar": limiar,
                "calibracao": metricas(y_calib, s_combinado_calib, limiar),
                "teste": m_teste,
                "teste_por_tecnica": por_tecnica.reset_index().to_dict(orient="records"),
                "teste_por_marcador": por_marcador.reset_index().to_dict(orient="records"),
                "baseline_para_comparacao": {"modelo": baseline_escolhido, **m_baseline},
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nResultados salvos em {ARQUIVO_SAIDA}")
