# pip install datasets pandas

from datasets import load_dataset
import pandas as pd
"""
Objetivo: carregar um conjunto de prompts legítimos reais.
"""
# Opção A: Stanford Alpaca (mais simples, bom para começar)
alpaca = load_dataset("tatsu-lab/alpaca")
df_normal = pd.DataFrame(alpaca["train"])
df_normal = df_normal[["instruction"]].rename(columns={"instruction": "text"})
df_normal["label"] = 0  # 0 = normal

# Opção B: LMSYS-Chat-1M (mais realista, dataset maior e mais pesado)
# lmsys = load_dataset("lmsys/lmsys-chat-1m")
# extrair só a primeira mensagem "user" de cada conversa

print(df_normal.shape)
df_normal.to_csv("data/prompts_normais.csv", index=False)