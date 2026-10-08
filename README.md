# Forensic Prompt Injection Detection

🇺🇸 [Read in English](README.en.md)

> 🚧 **Projeto em andamento.** Este repositório é o meu Projeto de Conclusão de Curso (PCC) e ainda está em desenvolvimento. Algumas etapas estão incompletas e ainda não há resultados finais.

Detecção de **prompt injection** em aplicações com LLM usando **detecção de anomalias sobre embeddings**. O detector aprende como é um prompt legítimo e sinaliza o que foge desse padrão, **antes** de o prompt chegar ao modelo de linguagem. Cada bloqueio fica registrado para análise forense.

## Ideia

Em vez de treinar um classificador com exemplos de ataques (que mudam o tempo todo), o projeto trata a injeção como **anomalia**:

1. Prompts legítimos são convertidos em embeddings.
2. Modelos de detecção de anomalia (**Isolation Forest** e **One-Class SVM**) são treinados **apenas com dados normais**.
3. Prompts maliciosos, gerados a partir das técnicas do **OWASP LLM01 (Prompt Injection)**, são usados só para calibrar o limiar de decisão e avaliar o detector.
4. Na aplicação final, o detector fica na frente da LLM e bloqueia prompts suspeitos, registrando score e horário para o relatório forense.

```
prompt ──► limpeza ──► embedding ──► detector de anomalia ──► score < limiar? ──► BLOQUEADO (+ registro forense)
                                                                    │
                                                                    └──► PERMITIDO ──► LLM
```

## Pipeline

| Etapa | Script | O que faz | Status |
|---|---|---|---|
| 1 | `Datasetbase.py` | Baixa o [Stanford Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) e monta os prompts legítimos (`instruction` + `input`) | ✅ |
| 2 | `PreProcessamento.py` | Limpa o texto e gera embeddings com `all-MiniLM-L6-v2` (384 dimensões) | ✅ |
| 3 | `ModeloDeteccaoAnomalias.py` | Treina Isolation Forest e One-Class SVM só com prompts normais; separa uma fatia para validação | ✅ |
| 4 | `aumentodedados.py` | Gera prompts maliciosos sintéticos com Mistral 7B (via Ollama) a partir de sementes do OWASP LLM01 e monta os conjuntos de calibração e teste | ✅ |
| 5 | `Avaliacao.py` | Matriz de confusão e métricas, priorizando o recall da classe maliciosa | 🚧 em andamento |
| 6 | — | Calibração do limiar de decisão usando o conjunto de calibração | ⏳ planejado |
| 7 | `TesteEndToEnd.py` | Simula a aplicação-alvo e mede se o detector intercepta o ataque antes da LLM | 🚧 esqueleto |

### Técnicas de ataque cobertas (etapa 4)

- **Role-play override**: o atacante pede que o modelo assuma uma persona sem restrições.
- **Hidden instruction**: uma instrução escondida dentro de uma tarefa legítima (tradução, resumo).
- **Payload splitting**: a instrução maliciosa é dividida em partes e remontada.
- **Indirect injection**: a instrução vem dentro de um conteúdo externo (e-mail, ticket).
- **Encoding**: o payload vem codificado em Base64 ou ROT13. A codificação é feita por código, não pela LLM, para garantir que seja válida.

As variações geradas passam por um filtro de similaridade, que descarta quase-duplicatas, e por uma checagem de fidelidade da técnica.
Os conjuntos de calibração (60%) e teste (40%) não compartilham nenhum exemplo, para evitar *data leakage*.

## Como executar

**Requisitos:** Python 3.11 (testado; 3.10+ deve funcionar) e, para a etapa 4, [Ollama](https://ollama.com/) com o modelo `mistral:7b` (`ollama pull mistral:7b`).

As versões exatas usadas estão em [`requirements.txt`](requirements.txt). Fixar as versões importa principalmente para o `scikit-learn`: os modelos salvos com `joblib` podem não carregar em outra versão.

```bash
# opcional: PyTorch só com CPU (mais leve)
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

Rode os scripts **a partir da raiz do repositório**, na ordem:

```bash
python Datasetbase.py
python PreProcessamento.py
python ModeloDeteccaoAnomalias.py
python aumentodedados.py        # precisa do Ollama rodando
```

Os dados gerados (CSVs, embeddings `.npy` e modelos `.joblib`) ficam em `data/`. Essa pasta não é versionada, porque tudo nela pode ser gerado de novo pelos scripts.

## Próximos passos

- [ ] Completar `Avaliacao.py` (carregar modelos e conjunto de teste; comparar os dois modelos)
- [ ] Calibrar o limiar de decisão no conjunto de calibração
- [ ] Completar o teste ponta a ponta com uma LLM local
- [ ] Gerar o registro forense dos prompts bloqueados
- [ ] Incluir exemplos reais de datasets públicos de prompt injection

## Créditos e licenças

- **Dados legítimos:** [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) (Taori et al., 2023), obtido via [Hugging Face](https://huggingface.co/datasets/tatsu-lab/alpaca), licença [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/): **somente uso não comercial**. O dataset não é redistribuído neste repositório; ele é baixado pelo `Datasetbase.py`.
- **Modelo de embeddings:** [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), licença Apache 2.0.
- **Geração de dados sintéticos:** [Mistral 7B](https://mistral.ai/) via [Ollama](https://ollama.com/), licença Apache 2.0.
- **Taxonomia de ataques:** [OWASP Top 10 for LLM Applications, LLM01: Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/).

O **código** deste repositório está sob a licença [MIT](LICENSE). A licença MIT vale só para o código: os dados gerados a partir do Alpaca continuam sujeitos à CC BY-NC 4.0.
