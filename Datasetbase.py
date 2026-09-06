# pip install datasets pandas

import os
from datasets import load_dataset
import pandas as pd
"""
Objetivo: carregar um conjunto de prompts legítimos reais.
"""
# Opção A: Stanford Alpaca (mais simples, bom para começar)
# O dataset tem 4 colunas: instruction, input, output, text.
# - instruction: o pedido do usuário
# - input: contexto opcional que acompanha o pedido (presente em ~40% das linhas)
# - output / text: resposta do modelo / template pronto -> não servem como "prompt"
# O prompt real enviado por um usuário é instruction + input (quando houver),
# então combinamos os dois em vez de usar só instruction.
alpaca = load_dataset("tatsu-lab/alpaca")
df_raw = pd.DataFrame(alpaca["train"])

df_normal = pd.DataFrame({
    "text": df_raw.apply(
        lambda row: row["instruction"] if row["input"] == ""
        else f"{row['instruction']}\n{row['input']}",
        axis=1,
    )
})
df_normal["label"] = 0  # 0 = normal

# Opção B: LMSYS-Chat-1M (mais realista, dataset maior e mais pesado)
# lmsys = load_dataset("lmsys/lmsys-chat-1m")
# extrair só a primeira mensagem "user" de cada conversa

print(df_normal.shape)
os.makedirs("data", exist_ok=True)
df_normal.to_csv("data/prompts_normais.csv", index=False)