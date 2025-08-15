# MLLM Vision Benchmarking

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)  
[![Hugging Face](https://img.shields.io/badge/Models-HuggingFace-yellow)](https://huggingface.co/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
---

This project benchmarks **Multimodal Large Language Models (MLLMs)** on fine-grained visual verification tasks — starting with **twin detection**, and eventually including **disguise detection**. It compares state-of-the-art models like **InternVL2.5**, **Qwen2.5-VL**, and **LLaVA-OneVision** (planned) against traditional **OpenCV baselines**.

---

## Project Overview

### Tasks
1. **Twin Verification**
   - Identify whether two face images depict the same person — including identical twins with subtle visual differences.

2. **Disguise Detection** *(Planned)*
   - Detect if a person is in disguise, wearing makeup, masks, or otherwise camouflaged.

---

## Pipeline

1. **Data Preparation**
   - CSV metadata files (`twin_face_pairs_absolute.csv`, `disguises_absolute.csv`) stored in `metadata/`
   - Face images stored in `data/twin_faces/` and `data/disguise_faces/`

2. **Data Loading**
   - `TwinDataset` & `DisguiseDataset` classes parse CSVs and load image pairs directly, with optional preprocessing.

3. **Model Inference**
   - Supports MLLMs via Hugging Face `transformers`.  
   - Pairwise comparison for twins — side-by-side combined images.  
   - Caption-based evaluation for disguised faces (planned).

4. **Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-score.  
   - Baseline comparison using OpenCV-based face matching.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/sihvenforucd/project-mllms_vision.git
cd project-mllms_vision

# Install required Python packages
pip install -r requirements.txt
```


---

## Usage

### 1. Launch on Google Colab
- Use the Colab badge above to open the `twin_benchmark.ipynb`.
- Mount Google Drive and select GPU runtime.

### 2. Load the Twin Dataset
```python
from twin_dataloader import TwinDataset

twin_dataset = TwinDataset(
    csv_path='/content/drive/MyDrive/vision_benchmark/metadata/twin_face_pairs_absolute.csv',
    root_dir='/content/drive/MyDrive/vision_benchmark/data/twin_faces'
)
```
### 3. Run Benchmark
```python
from multi_mllm_runner import evaluate_twin_model

results = evaluate_twin_model(
    model_id="OpenGVLab/InternVL-Chat-V1-2",
    twin_pairs=[(label, img1, img2) for label, img1, img2 in twin_dataset]
)
```

### Save Results
```python
import pandas as pd
pd.DataFrame(results).to_csv('twin_internvl_results.csv', index=False)
```

## Directory Structure
project-mllms_vision/
│
├── data/
│   ├── twin_faces/
│   └── disguise_faces/
│
├── metadata/
│   ├── twin_face_pairs_absolute.csv
│   ├── disguises_absolute.csv
│
├── notebooks/
│   └── twin_benchmark.ipynb
│
├── twin_dataloader.py
├── disguise_dataloader.py
├── multi_mllm_runner.py
├── opencv_twin_baseline.py
└── metrics_logger.py

## Models Supported
- InternVL-Chat-V1-2 (InternVL2.5) — currently implemented

- Qwen2.5-VL (planned)

- LLaVA-OneVision (planned)

- MiniGPT-4 (planned)


