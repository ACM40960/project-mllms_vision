
# 🔍 VISBENCH-3: Benchmarking MLLMs on Fine-Grained Visual Reasoning Tasks

VISBENCH-3 is a comprehensive benchmark for evaluating **Multimodal Large Language Models (MLLMs)** on challenging fine-grained visual reasoning tasks:

- 👯‍♀️ **Twin Face Verification**
- 🥸 **Disguise Detection**
- 🐾 **Wildlife Species Recognition (Night-vision)**

---

## ✨ Highlights

- 📊 Benchmarked **Qwen2.5-VL**, **LLaMA-4 Maverick**, **LLaMA-4 Scout** across 3 diverse tasks
- 📈 Evaluated using **Accuracy**, **F1 Score**, and **Robustness**
- 🧠 Included architectural analysis for each model
- 📦 Open-sourced pipeline for replication and extension

---

## 🏗️ Architecture Overview

### 🔹 [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct)
- Developed by Alibaba.
- Based on the **Transformer decoder** with **Vision and Language fusion** at token-level.
- Accepts **images and text jointly** via a vision encoder + projection to LLM tokens.

### 🔸 [LLaMA-4 Maverick](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8)
- Meta’s open-weight LLM with **vision support** using embedded visual tokens.
- Uses **dual-encoder fusion** between vision encoder and text decoder.
- Fine-tuned for **multimodal instruction-following**.

### 🔸 [LLaMA-4 Scout](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct)
- An enhanced LLaMA variant for **instruction-tuned visual reasoning**.
- Integrates attention mechanisms across image patches + text.
- Performs better on **binary vision classification** tasks (like twin/disguise).

---

## 📊 Evaluation Visuals

### 🔹 Dataset Difficulty Ranking
![Dataset Difficulty](assets/plot1_difficulty.png)

> Higher F1 → Easier for the model. Twin is easier than Wildlife.

---

### 🔹 Radar Plot: Overall Model Performance
![Radar All Models](assets/plot2_radar.png)

> LLaMA-Scout consistently ranks highest across all metrics.

---

### 🔹 F1 Score Trend by Dataset
![F1 Robustness](assets/plot3_f1_trend.png)

> Qwen2.5 degrades more gracefully across difficult datasets.

---

### 🔹 Accuracy Comparison
![Accuracy Bar](assets/plot4_accuracy.png)

> Scout outperforms on disguise detection; Maverick is stronger on wildlife.

---

### 🔹 F1 Score Comparison
![F1 Bar](assets/plot5_f1.png)

---

### 🔹 F1 Table Summary
![F1 Table](assets/plot6_table.png)

> Best average: **LLaMA-Scout (0.6032)**

---

### 🔹 Best Model Per Dataset (Pie Chart)
![Best Model Pie](assets/plot7_pie.png)

> LLaMA-Scout wins on 2 out of 3 datasets.

---

## 🧪 Dataset Tasks

| Task       | Description                                     |
|------------|--------------------------------------------------|
| Twin       | Classify whether face pairs are real twins       |
| Disguise   | Detect if one image is a disguised version       |
| Wildlife   | Classify animal species from IR night images     |

---

## 🚀 Quickstart

```bash
git clone https://github.com/your-username/VISBENCH-3.git
cd VISBENCH-3
pip install -r requirements.txt
python run_benchmark.py --model Qwen2.5-VL --task disguise
```

---

## 📁 Repository Structure

```
├── assets/                     # 📸 All plots and diagrams
├── data/                       # 📂 Datasets for 3 tasks
├── notebooks/                  # 🧠 Evaluation scripts
├── src/                        # 🔧 Core benchmark engine
└── README.md                   # 📘 This file
```

---

## 📈 Future Work

- Expand to **video QA**, **multi-turn image chat**, and **VQA reasoning**
- Add evaluation under **distribution shift** and **real-world noise**
- Introduce **human-in-the-loop verification** for borderline cases

---

## 👨‍🔬 Authors

- **Kiara** — Principal Researcher | UCD MSc | Vision-Language Systems
- **Smart Optimization** — Project Lead & Benchmark Design

---

## 📄 License

MIT License

---

## 📚 Citation

```bibtex
@misc{visbench3,
  title={VISBENCH-3: Fine-Grained Vision Reasoning Benchmark for MLLMs},
  author={Kiara, Smart Optimization},
  year={2025},
  url={https://github.com/your-username/VISBENCH-3}
}
```
