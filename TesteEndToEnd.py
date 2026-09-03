"""
Objetivo: simular a aplicação-alvo e medir se o detector intercepta antes da LLM processar.
"""
# Estrutura conceitual do teste

def pipeline_defesa(prompt: str, detector, modelo_embedding, limiar):
    embedding = modelo_embedding.encode([prompt])
    score = detector.decision_function(embedding)[0]

    if score < limiar:
        return {
            "status": "BLOQUEADO",
            "score": score,
            "prompt": prompt,
            "timestamp": ...,  # para o relatório forense
        }
    else:
        # só aqui o prompt "chegaria" na LLM local
        # resposta = llm_local.generate(prompt)
        return {
            "status": "PERMITIDO",
            "score": score,
            "prompt": prompt,
        }