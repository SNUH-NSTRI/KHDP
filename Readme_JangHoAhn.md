# 🏥 SNUH-Note Infectious Disease RAG Pipeline 
> Author: Jang Ho Ahn
>
> End-to-end workflow for building a **domain-specialised QA model** that answers
interdepartmental infectious disease consultations by combining *LoRA-based supervised fine-tuning* with *Retrieval-Augmented Generation (RAG)*.
---

## 📑 Project Overview (main_JangHoAhn.ipynb)
| Stage | Description |
|-------|-------------|
| 1️⃣ Data Extraction | Pull infectious-disease consultation notes (SNUH NOTE) that match *mdfm_id* `41813`, `41814`, `42345`. |
| 2️⃣ QA Construction | Treat the **request** from the attending physician as the **Question** and the **reply** from the infectious-disease specialist as the **Answer** (290 matched pairs). |
| 3️⃣ Translation | Translate **Answer** text to English with `facebook/nllb-200-3.3B`. |
| 4️⃣ Fine-Tuning | LoRA SFT on the 9:1 train/test split using `Intelligent-Internet/II-Medical-8B-1706` as the base. A lightweight architecture was selected and quantized to optimize performance on the **Tesla T4** environment. |
| 5️⃣ External Knowledge | Collect 100 UpToDate PDF articles (under the "Infectious Diseases" category, focusing on high-prevalence local diseases and management-related topics) → split to chunks → embed with **MedCPT Query Encoder** (pre-trained on 255 M PubMed pairs) → index with **FAISS**. |
| 6️⃣ RAG Inference | 3-step retrieval (Dense → Cross-Encoder re-rank → MMR) to collect evidence, then generate the answer with the fine-tuned model. |
| 7️⃣ Evaluation | Metrics on the held-out 10 % test set: **Mean Cosine Similarity**, ROUGE-LCS F1, BLEU. |

---

## 🗂 Folder Structure
```
.
├── rag/ # 100 UpToDate PDFs (infectious diseases)
├── datasets/
│ └── SNUHNOTE/ # Raw EMR Excel files
├── main_JangHoAhn.ipynb # Main end-to-end notebook
└── requirements_JangHoAhn.txt
```
---

## 🔧 Key Components

### 1. Data Mining & Filtering
* **Keywords**: “승인”, “문의”, “추천”, “권장”, “고려”, “니다”, “권고”.
* Remove boilerplate text (e.g., variations of “승인하였습니다.”) and canceled orders (e.g., “취소된 오더입니다.”).
* Group by `nid` and merge consecutive notes from the same day into a single record.

### 2. QA Pair Generation
```text
Question  → Attending Physician’s consultation note  
Answer    → Infectious disease specialist’s reply (English-translated)
```
Total: **290** pairs → **261 train / 29 test** after 9:1 split.

### 3. Model Fine-Tuning
* **Base**: `Intelligent-Internet/II-Medical-8B-1706` (46.8 % HealthBench).
* **Method**: LoRA SFT on train set.
* **Output**: `lora_rag_weight` adapter.

### 4. Retrieval-Augmented Generation (3-step retrieval)
* **Retriever** – `AdvancedRetriever` executes a *three-stage* pipeline  
  1. **Stage-1 · Dense search** Embed query & chunks with **MedCPT** and fetch the top *recall_k* hits from the **FAISS** index.  
  2. **Stage-2 · Re-ranking** Score each (query, chunk) pair with the cross-encoder `cross-encoder/ms-marco-MiniLM-L6-v2`; keep the best *rerank_k*.  
  3. **Stage-3 · MMR diversification (optional)** Apply **Maximal Marginal Relevance** to the reranked set to remove near-duplicates while preserving relevance.

* **Embedder** – `ncbi/MedCPT-Query-Encoder` (768-d).  
* **Index** – FAISS `IndexFlatIP`.  
* **Prompt Template**
  ```
  @@@ Evidence
  {retrieved_chunks}

  @@@ Instruction: Base your [@@@ Answer] ONLY on the evidence above …
  {user_question}
  ```
* **Generator** – LoRA-fine-tuned `II-Medical-8B-1706` with decoding tuned to minimise n-gram repetition.

### 5. Metrics & Results
| Metric | Test-set Score | 
|--------|----------------|
| Mean Cosine Similarity | **0.7695** |
---

## 🚀 Quick Start

```bash
# 1) Install
conda create -n snuh_rag python=3.10
conda activate snuh_rag
pip install -r requirements.txt

# 2) Run notebook
jupyter lab main_JangHoAhn.ipynb
```

Inside the notebook, execute cells sequentially:
1. **Settings** – libraries & helper functions  
2. **Finetuning** – extract notes → build QA → LoRA SFT  
3. **RAG** – embed PDFs, build FAISS index  
4. **Evaluation** – compute metrics, view sample outputs  

---

## 📌 References
* **LoRA**: Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021  
* **MedCPT**: Jin et al., *MedCPT: A Pre-trained Model for Biomedical Retrieval*, 2023  
* **UpToDate**: Wolters Kluwer Health, *Clinical decision support resource*  

---

## 📜 License
This code is for research and educational purposes.  
Please ensure compliance with the licenses of the underlying models (II-Medical-8B-1706, MedCPT, NLLB-200) and source data (UpToDate articles, SNUH Note).

*Vital-Lab Datathon 2025*
