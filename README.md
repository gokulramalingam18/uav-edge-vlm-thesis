# UAV Edge VLM — MSc Thesis

**Resource-Efficient Multimodal Foundation Models for Real-Time UAV Perception on Edge Devices**

*Gokul Ramalingam — University of Twente, MSc Embedded Systems, 2026*

---

## What This Is

This repository contains the full implementation of my MSc thesis: fine-tuning a vision-language foundation model (Qwen2.5-VL-3B) for UAV aerial scene understanding, and deploying it on a resource-constrained edge device (NVIDIA Jetson Orin NX 16GB) — without cloud connectivity.

The model takes a drone image as input and outputs a structured natural-language inventory of detected objects and scene type, running entirely on-device.

---

## Key Results

### Pareto Frontier (latency vs. accuracy on Jetson Orin NX)

| Configuration | Latency | MAE | Model Size | Notes |
|---|---|---|---|---|
| f16, 1024 tokens (A) | 9.39s | 14.27 | 5.8 GB | full precision baseline |
| f16, 512 tokens (B) | 6.79s | 17.78 | 5.8 GB | faster, less accurate |
| f16, 256 tokens (C) | 5.75s | 16.23 | 5.8 GB | fastest f16 |
| **Q4+imatrix, 1024 tokens (D ★)** | **8.10s** | **14.10** | **1.8 GB** | **recommended** |

**Point D (Q4+imatrix) is Pareto-dominant over Point A** — faster, more accurate, and 3.2× smaller. Naive Q4/Q8 quantization produced garbage output; importance-matrix calibration with density-stratified calibration data rescued the fine-tuned adapter.

### Fine-Tuning Results (648 held-out VisDrone images)

| Metric | Value |
|---|---|
| Format compliance | 99% (641/648 parsed) |
| Mean absolute error (MAE) | 16.79 |
| Median absolute error | 11 |
| Trimmed MAE (worst 5% removed) | 13.73 |
| 90th-percentile error | 38 |

---

## Key Findings

**1. Deployment architecture dominates latency more than the model.**
Naive per-frame model reload: 32.5s. Persistent server (load once, stream frames): 9.4s.
69% of measured latency was reload overhead — not computation.

**2. Ollama's import layer corrupts fine-tuned Qwen2.5-VL models.**
Proven via controlled experiment: vision weights are bit-identical between base and merged model (max diff = 0.0 across 390 tensors), and the same GGUF files produce correct output through native llama.cpp but garbage through Ollama. The fault is isolated specifically to Ollama's import path for this model class.

**3. imatrix-calibrated quantization rescues fine-tuned adapters.**
Naive Q4/Q8 erases LoRA deltas (they fall below the rounding resolution). imatrix assigns higher precision to critical weights before rounding, preserving fine-tuned behavior. Calibration data was density-stratified (70% dense scenes) to target the model's known failure mode.

**4. Dense-scene counting degrades systematically.**
Over-counting bias peaks at moderate density (GT 40-60, bias +14.3) and drops at extreme density (GT 80+, bias +6.0) where the model falls back on training-data associations, under-counting scenes with 100+ objects. Fix: add more extreme-density training examples.

**5. JetPack 5.1.3 blocks the Python/transformers route entirely.**
Python 3.8 is locked to the only NVIDIA CUDA PyTorch wheel available for this JetPack version. transformers ≥4.49 (needed for Qwen2.5-VL) requires Python ≥3.9. Solution: bypass Python entirely — compile llama.cpp natively on the Jetson and serve via persistent HTTP server.

---

## Hardware & Software

| Component | Details |
|---|---|
| Edge device | NVIDIA Jetson Orin NX 16GB, JetPack 5.1.3 |
| Training hardware | UT EWI HPC cluster (SLURM, main-gpu partition) |
| Base model | Qwen2.5-VL-3B-Instruct (Alibaba) |
| Dataset | VisDrone2019-DET → 5,823 UAV captions |
| Fine-tuning | QLoRA, r=8, α=16, 1,000 samples, 3 epochs, bf16 |
| Inference runtime | llama.cpp (native build, CUDA SM 8.7) |
| Quantization | Q4_K_M + imatrix calibration |

---

## Repository Structure

```
scripts/hpc/          Training and evaluation scripts (run on HPC cluster)
  finetune.py         QLoRA fine-tuning with answer-only masking
  evaluate_finetuned.py  648-image quantitative evaluation
  compare_models.py   Base vs fine-tuned qualitative comparison
  convert_visdrone.py VisDrone detection annotations → captions
  export_gguf.py      Merge adapter + convert to GGUF
  plot_eval.py        Error distribution figures

scripts/jetson/       Inference and benchmarking (run on Jetson)
  pareto_eval.py      Latency + accuracy benchmark across token settings
  bench_server.py     Load-once persistent server benchmark
  make_demo.py        Annotated demo image generation (GT vs prediction)
  annotate_video.py   UAV footage annotation pipeline

slurm/                SLURM job files for HPC
  job.slurm           Fine-tuning job
  imatrix.slurm       imatrix calibration job
  quantize_imatrix.slurm  Calibrated quantization job

results/              Experiment outputs
  eval_results.csv    648-image evaluation (gt, pred, error per image)
  pareto_*.csv        Per-image results for each Pareto point
  figures/            Plots (Pareto frontier, error distribution)

docs/
  IMPLEMENTATION_HANDOFF.md  Environment setup, gotchas, resume point
```

---

## Reproduction

Model weights are not included (too large for GitHub). To reproduce:

1. Download `Qwen/Qwen2.5-VL-3B-Instruct` from HuggingFace
2. Download VisDrone2019-DET-train dataset
3. Generate captions: `python scripts/hpc/convert_visdrone.py`
4. Fine-tune: `sbatch slurm/job.slurm`
5. Merge + convert: `python scripts/hpc/export_gguf.py`
6. Calibrate: `sbatch slurm/imatrix.slurm`
7. Quantize: `sbatch slurm/quantize_imatrix.slurm`
8. Deploy on Jetson: see `docs/IMPLEMENTATION_HANDOFF.md`

---

## Notes

The main unexpected finding was that Q4+imatrix outperformed f16 on both latency and accuracy simultaneously — making quantization a strict improvement rather than a trade-off for this setup. The Ollama corruption bug (which took several days to root-cause) turned out to be cleanly isolatable and is documented in detail in the thesis.

*Thesis defense: August 2026, University of Twente*
