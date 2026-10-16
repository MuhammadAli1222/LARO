# LARO: Learnable Adaptive Retrieval Orchestration for Latency-Efficient and Grounded LLMs

## 📌 Overview
Retrieval-Augmented Generation (RAG) models suffer from an **Accuracy–Latency Bottleneck**:
* **Over-retrieving** for simple queries incurs redundant latency.
* **Under-retrieving** for complex queries causes hallucinations.
* Existing active retrieval baselines (FLARE, DRAGIN) execute 3.0–3.5 decoding pauses per query, disrupting Key-Value (KV) cache pipelining.

**LARO** resolves this by treating retrieval as a resource-constrained sequential decision problem:
1. **Single-Pass Epistemic Sensing:** Extracts 4 uncertainty signals (σ_ppl, σ_attn, σ_disp, M(q)) during the initial prompt forward pass..
2. **Tier 1 Parametric Fast Path:** Bypasses retrieval ($k=0$) in **0.72s** when model confidence is high (resolving 53.15% of queries).
3. **Tier 2 Learned Policy Controller:** A 3-layer neural MLP ($388 \to 4$) trained via Oracle-Guided Behavioral Cloning to dynamically select between **Lucene BM25**, **FAISS Dense**, or **Query Reformulation**, dynamically scaling depth $k \in \{0, 1, 5, 10\}$.

LARO slashes context token overhead by an **effective 76.58% (Sparsity Saved)** while achieving a **1.51×–2.10× latency speedup** over Vanilla RAG.
---

## 🛠️ Environment Setup

```bash
# 1. Clone the repository
git clone https://github.com/MuhammadAli1222/LARO.git
cd LARO


# 2. Create conda environment (Python 3.10)
conda create -n laro_env python=3.10 -y
conda activate laro_env

# 3. Install PyTorch with CUDA 12.1
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# 4. Install dependencies
torch>=2.2.0
transformers>=4.40.0
bitsandbytes>=0.43.0
accelerate>=0.29.0
sentence-transformers>=2.6.0
faiss-gpu>=1.8.0
pyserini>=0.22.0
scikit-learn>=1.4.0
scipy>=1.12.0
tqdm>=4.66.0
