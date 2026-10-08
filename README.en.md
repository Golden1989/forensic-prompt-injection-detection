# Forensic Prompt Injection Detection

🇧🇷 [Leia em português](README.md)

> 🚧 **Work in progress.** This repository is my undergraduate capstone project and is still under development. Some stages are incomplete and there are no final results yet.

Detecting **prompt injection** in LLM applications through **anomaly detection over embeddings**. The detector learns what a legitimate prompt looks like and flags anything that deviates from that pattern **before** the prompt reaches the language model. Every block is logged for forensic analysis.

## Approach

Attacks change all the time, so instead of training a classifier on attack examples, this project treats injection as an **anomaly**:

1. Legitimate prompts are converted into embeddings.
2. Anomaly detection models (**Isolation Forest** and **One-Class SVM**) are trained **on normal data only**.
3. Malicious prompts, generated from the techniques in **OWASP LLM01 (Prompt Injection)**, are used only to calibrate the decision threshold and evaluate the detector.
4. In the final application, the detector sits in front of the LLM and blocks suspicious prompts, logging score and timestamp for the forensic report.

```
prompt ──► cleaning ──► embedding ──► anomaly detector ──► score < threshold? ──► BLOCKED (+ forensic log)
                                                                 │
                                                                 └──► ALLOWED ──► LLM
```

## Pipeline

| Stage | Script | What it does | Status |
|---|---|---|---|
| 1 | `Datasetbase.py` | Downloads [Stanford Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) and builds the legitimate prompts (`instruction` + `input`) | ✅ |
| 2 | `PreProcessamento.py` | Cleans the text and generates embeddings with `all-MiniLM-L6-v2` (384 dimensions) | ✅ |
| 3 | `ModeloDeteccaoAnomalias.py` | Trains Isolation Forest and One-Class SVM on normal prompts only; holds out a validation split | ✅ |
| 4 | `aumentodedados.py` | Generates synthetic malicious prompts with Mistral 7B (via Ollama) from OWASP LLM01 seeds and builds the calibration and test sets | ✅ |
| 5 | `Avaliacao.py` | Confusion matrix and metrics, prioritizing recall on the malicious class | 🚧 in progress |
| 6 | — | Decision threshold calibration using the calibration set | ⏳ planned |
| 7 | `TesteEndToEnd.py` | Simulates the target application and checks whether the detector intercepts the attack before the LLM | 🚧 skeleton |

> File names and code comments are in Portuguese.

### Attack techniques covered (stage 4)

- **Role-play override**: the attacker asks the model to adopt an unrestricted persona.
- **Hidden instruction**: an instruction hidden inside a legitimate task (translation, summarization).
- **Payload splitting**: the malicious instruction is split into parts and reassembled.
- **Indirect injection**: the instruction comes inside external content (an email, a support ticket).
- **Encoding**: the payload is Base64- or ROT13-encoded. Encoding is done in code, not by the LLM, so it is always valid.

Generated variations go through a similarity filter that drops near-duplicates, and a technique-fidelity check.
The calibration (60%) and test (40%) sets share no examples, to avoid data leakage.

## Running it

**Requirements:** Python 3.10+ and, for stage 4, [Ollama](https://ollama.com/) with the `mistral:7b` model (`ollama pull mistral:7b`).

```bash
pip install datasets pandas numpy scikit-learn sentence-transformers joblib requests
```

Run the scripts **from the repository root**, in order:

```bash
python Datasetbase.py
python PreProcessamento.py
python ModeloDeteccaoAnomalias.py
python aumentodedados.py        # requires Ollama running
```

Generated data (CSVs, `.npy` embeddings and `.joblib` models) goes into `data/`. That folder is not versioned, since the scripts can regenerate everything in it.

## Next steps

- [ ] Finish `Avaliacao.py` (load models and test set; compare both models)
- [ ] Calibrate the decision threshold on the calibration set
- [ ] Complete the end-to-end test with a local LLM
- [ ] Produce the forensic log of blocked prompts
- [ ] Add real examples from public prompt injection datasets
