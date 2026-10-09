"""
Objetivo: medir performance, priorizando recall da classe anômala.

Fluxo, em três partes que nunca se misturam:
  1. Scores: cada modelo dá um "score de anomalia" para cada prompt.
  2. Calibração: usando SÓ o conjunto de calibração, escolhe o modelo e o
     limiar de decisão (score acima do limiar = BLOQUEADO).
  3. Teste: aplica o modelo e o limiar já congelados ao conjunto de teste,
     uma única vez, e reporta as métricas finais.

O conjunto de teste nunca influencia nenhuma escolha. Testes externos (ex.:
ataques reais de datasets públicos) entram depois pela função avaliar(),
com o mesmo limiar congelado.

Uso:
    python Avaliacao.py
"""
import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")  # só salva arquivos, não abre janela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from aumentodedados import limpar_texto

# Quanto dos ataques de calibração o limiar precisa pegar. Em segurança,
# deixar passar um ataque costuma custar mais que um alarme falso, então o
# critério prioriza recall; entre os limiares que atingem a meta, fica o de
# maior precisão (menos alarmes falsos). Fixado ANTES de olhar o teste.
RECALL_ALVO = 0.95

MODELOS = {
    "IsolationForest": "data/modelo_isolation_forest.joblib",
    "OneClassSVM": "data/modelo_ocsvm.joblib",
}

PASTA_RESULTADOS = "resultados"


def score_anomalia(modelo, X: np.ndarray) -> np.ndarray:
    """decision_function devolve valores positivos para "normal" e negativos
    para "anômalo". Invertemos o sinal para ter um score em que MAIOR = mais
    suspeito, que é a convenção das métricas do scikit-learn (a classe
    positiva, 1, é a maliciosa)."""
    return -modelo.decision_function(X)


def metadados_maliciosos(conjunto: str) -> pd.DataFrame:
    """Devolve técnica e marcador_ok de cada malicioso do conjunto, na MESMA
    ordem em que aparecem nos arrays .npy.

    Os .npy não guardam essa informação, mas a divisão feita em
    aumentodedados.py é determinística (random_state=42): refazendo-a aqui,
    recuperamos a ordem. Para não confiar só nisso, recalcula os embeddings
    e confere que batem com os salvos.
    """
    df = pd.read_csv("data/prompts_maliciosos.csv")
    idx_calib, idx_teste = train_test_split(
        np.arange(len(df)), test_size=0.4, random_state=42, stratify=df["tecnica"]
    )
    idx = idx_teste if conjunto == "teste" else idx_calib
    meta = df.loc[idx, ["text", "tecnica", "marcador_ok"]].reset_index(drop=True)

    X = np.load(f"data/{'test' if conjunto == 'teste' else 'calibracao'}_embeddings.npy")
    y = np.load(f"data/{'test' if conjunto == 'teste' else 'calibracao'}_labels.npy")
    modelo_emb = SentenceTransformer("all-MiniLM-L6-v2")
    emb = modelo_emb.encode(meta["text"].map(limpar_texto).tolist(), batch_size=32)
    if not np.allclose(X[y == 1], emb, atol=1e-5):
        raise RuntimeError(
            "A ordem dos maliciosos não bate com os .npy. O aumentodedados.py "
            "foi rodado de novo depois de gerar os conjuntos?"
        )
    return meta


def escolher_limiar(y: np.ndarray, scores: np.ndarray, recall_alvo: float) -> float:
    """Entre os limiares que atingem recall >= recall_alvo, escolhe o de
    maior precisão."""
    precisao, recall, limiares = precision_recall_curve(y, scores)
    # precisao e recall têm um elemento a mais que limiares (o ponto final)
    validos = np.where(recall[:-1] >= recall_alvo)[0]
    melhor = validos[np.argmax(precisao[validos])]
    return float(limiares[melhor])


def metricas(y: np.ndarray, scores: np.ndarray, limiar: float) -> dict:
    """Métricas de um modelo num conjunto, com um limiar fixo."""
    pred = (scores >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precisao = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "recall": recall,  # dos ataques, quantos foram bloqueados
        "precisao": precisao,  # dos bloqueios, quantos eram ataque
        "f1": 2 * precisao * recall / (precisao + recall) if precisao + recall else 0.0,
        "taxa_falso_positivo": fp / (fp + tn),  # dos normais, quantos foram bloqueados
        "pr_auc": average_precision_score(y, scores),  # não depende do limiar
        "roc_auc": roc_auc_score(y, scores),  # não depende do limiar
        "matriz": {"VN": int(tn), "FP": int(fp), "FN": int(fn), "VP": int(tp)},
    }


def imprimir(nome: str, m: dict) -> None:
    mz = m["matriz"]
    print(
        f"  {nome:16s} recall {m['recall']:.3f} | precisão {m['precisao']:.3f} | "
        f"F1 {m['f1']:.3f} | falso positivo {m['taxa_falso_positivo']:.3%} | "
        f"PR-AUC {m['pr_auc']:.3f} | ROC-AUC {m['roc_auc']:.3f}"
    )
    print(f"  {'':16s} VP {mz['VP']}  FN {mz['FN']}  FP {mz['FP']}  VN {mz['VN']}")


def recall_por_grupo(scores_mal: np.ndarray, limiar: float, grupos: pd.Series) -> pd.DataFrame:
    """Recall separado por grupo (técnica, marcador...), só sobre os maliciosos."""
    pegos = scores_mal >= limiar
    tabela = pd.DataFrame({"grupo": grupos.values, "pego": pegos})
    return tabela.groupby("grupo")["pego"].agg(total="size", bloqueados="sum", recall="mean")


def avaliar(nome: str, X: np.ndarray, y: np.ndarray, modelo, limiar: float) -> dict:
    """Aplica modelo e limiar congelados a um conjunto de teste qualquer.
    É o ponto de entrada para testes externos no futuro."""
    m = metricas(y, score_anomalia(modelo, X), limiar)
    print(f"\n=== {nome} ===")
    imprimir("", m)
    return m


def salvar_graficos(scores_teste: dict, y_teste: np.ndarray, escolhido: str, limiar: float) -> None:
    """Curva precisão-recall dos dois modelos e distribuição dos scores do
    modelo escolhido, com o limiar marcado."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for nome, s in scores_teste.items():
        p, r, _ = precision_recall_curve(y_teste, s)
        ax.plot(r, p, label=f"{nome} (PR-AUC {average_precision_score(y_teste, s):.3f})")
    ax.axhline(y_teste.mean(), color="grey", linestyle=":", label=f"aleatório ({y_teste.mean():.3f})")
    ax.set_xlabel("Recall (ataques bloqueados)")
    ax.set_ylabel("Precisão (bloqueios corretos)")
    ax.set_title("Curva precisão-recall — conjunto de teste")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{PASTA_RESULTADOS}/curva_pr_teste.png", dpi=150)
    plt.close(fig)

    s = scores_teste[escolhido]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bins = np.linspace(s.min(), s.max(), 60)
    ax.hist(s[y_teste == 0], bins=bins, alpha=0.6, density=True, label="normal")
    ax.hist(s[y_teste == 1], bins=bins, alpha=0.6, density=True, label="malicioso")
    ax.axvline(limiar, color="black", linestyle="--", label="limiar")
    ax.set_xlabel("Score de anomalia (maior = mais suspeito)")
    ax.set_ylabel("Densidade")
    ax.set_title(f"Distribuição dos scores — {escolhido}, teste")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{PASTA_RESULTADOS}/distribuicao_scores_teste.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)

    modelos = {nome: joblib.load(caminho) for nome, caminho in MODELOS.items()}
    X_calib = np.load("data/calibracao_embeddings.npy")
    y_calib = np.load("data/calibracao_labels.npy").astype(int)
    X_teste = np.load("data/test_embeddings.npy")
    y_teste = np.load("data/test_labels.npy").astype(int)
    print(f"Calibração: {len(y_calib)} prompts ({y_calib.sum()} maliciosos, {y_calib.mean():.1%})")
    print(f"Teste:      {len(y_teste)} prompts ({y_teste.sum()} maliciosos, {y_teste.mean():.1%})")

    # ---------- 1. Scores ----------
    scores_calib = {nome: score_anomalia(m, X_calib) for nome, m in modelos.items()}
    scores_teste = {nome: score_anomalia(m, X_teste) for nome, m in modelos.items()}

    # ---------- 2. Calibração ----------
    # Referência: o limiar "de fábrica" de cada modelo (decision_function = 0),
    # que vem do contamination/nu = 0.05 usado no treino.
    print("\n=== CALIBRAÇÃO — limiar padrão do modelo (score 0) ===")
    for nome, s in scores_calib.items():
        imprimir(nome, metricas(y_calib, s, 0.0))

    print(f"\n=== CALIBRAÇÃO — limiar escolhido para recall >= {RECALL_ALVO:.0%} ===")
    limiares = {}
    for nome, s in scores_calib.items():
        limiares[nome] = escolher_limiar(y_calib, s, RECALL_ALVO)
        imprimir(nome, metricas(y_calib, s, limiares[nome]))
        print(f"  {'':16s} limiar = {limiares[nome]:.4f}")

    # O modelo é escolhido pela PR-AUC na calibração: ela resume o modelo em
    # todos os limiares e é a métrica adequada quando a classe positiva é rara.
    pr_auc_calib = {nome: average_precision_score(y_calib, s) for nome, s in scores_calib.items()}
    escolhido = max(pr_auc_calib, key=pr_auc_calib.get)
    limiar = limiares[escolhido]
    print(f"\nModelo escolhido (maior PR-AUC na calibração): {escolhido}, limiar {limiar:.4f}")

    # ---------- 3. Teste (uma única vez) ----------
    print("\n=== TESTE — modelos e limiares congelados da calibração ===")
    resultados_teste = {}
    for nome, s in scores_teste.items():
        resultados_teste[nome] = metricas(y_teste, s, limiares[nome])
        imprimir(nome + (" *" if nome == escolhido else ""), resultados_teste[nome])
    print("  (* = modelo escolhido)")

    meta_teste = metadados_maliciosos("teste")
    s_mal = scores_teste[escolhido][y_teste == 1]
    por_tecnica = recall_por_grupo(s_mal, limiar, meta_teste["tecnica"])
    print(f"\n=== TESTE — recall por técnica ({escolhido}) ===")
    print(por_tecnica.to_string(float_format=lambda v: f"{v:.3f}"))

    # Só hidden e indirect têm marcador; nas outras técnicas marcador_ok é
    # sempre True e não diz nada.
    com_marcador = meta_teste["tecnica"].isin(["hidden instruction", "indirect injection"]).values
    por_marcador = recall_por_grupo(
        s_mal[com_marcador],
        limiar,
        meta_teste.loc[com_marcador, "marcador_ok"].map({True: "com marcador", False: "naturalizado"}),
    )
    print(f"\n=== TESTE — hidden + indirect: óbvio vs. naturalizado ({escolhido}) ===")
    print(por_marcador.to_string(float_format=lambda v: f"{v:.3f}"))

    # ---------- Salvar ----------
    salvar_graficos(scores_teste, y_teste, escolhido, limiar)
    with open(f"{PASTA_RESULTADOS}/limiar_escolhido.json", "w", encoding="utf-8") as f:
        json.dump(
            {"modelo": escolhido, "arquivo_modelo": MODELOS[escolhido], "limiar": limiar, "recall_alvo": RECALL_ALVO},
            f,
            indent=2,
            ensure_ascii=False,
        )
    with open(f"{PASTA_RESULTADOS}/avaliacao_sintetica.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "recall_alvo": RECALL_ALVO,
                "modelo_escolhido": escolhido,
                "pr_auc_calibracao": pr_auc_calib,
                "limiares": limiares,
                "teste": resultados_teste,
                "teste_por_tecnica": por_tecnica.reset_index().to_dict(orient="records"),
                "teste_por_marcador": por_marcador.reset_index().to_dict(orient="records"),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nResultados salvos em {PASTA_RESULTADOS}/")
