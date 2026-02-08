# [ICLR 2026] D\&R: Recovery-based AI-Generated Text Detection via a Single Black-box LLM Call

<div align="center">

[![Conference](https://img.shields.io/badge/ICLR-2026-blue)](https://iclr.cc/)

**A novel, robust, and effective framework for detecting AI-generated text using the principle of posterior concentration.**


</div>

---

## 📢 News
- **[2026.01]** 🎉 Our paper has been accepted to **ICLR 2026**!
- **[2025.09]** Initial code release.

---

## 🚀 Introduction

We introduce **Disrupt & Recover (D&R)**, a recovery-based detection framework built on the principle of **posterior concentration**. Unlike traditional methods that rely on white-box features or specialized classifiers, D&R leverages a single black-box LLM call to differentiate between human-written and AI-generated text.

### Key Features
- **💡 Single Black-box Call**: Efficient detection without requiring access to model gradients or logits.
- **🛡️ Robustness**: Maintains high performance even under source-recovery mismatch and model variations.
- **🏆 SOTA Performance**: Outperforms existing baselines on both long and short text generation benchmarks.
- **⚡ Interpretability**: Built on solid theoretical ground (posterior concentration).


---

## 📊 Results  

Extensive experiments on **four datasets** and **six source models** show that D&R achieves **state-of-the-art detection performance**:  

- **AUROC 0.96** on long texts  
- **AUROC 0.87** on short texts  
- Gains of **+0.08** (long) and **+0.14** (short) compared to the strongest baseline  

D&R remains robust under **source–recovery mismatch** and **model variation**, making it broadly applicable.  

---

## 📦 Installation
```bash
git clone
pip install -r requirements.txt
```

---

## ℹ️ Note

This repository is actively under development. We have uploaded the majority of the core code, and additional modules, documentation, and refinements will be released as soon as possible. Thank you for your understanding and support.