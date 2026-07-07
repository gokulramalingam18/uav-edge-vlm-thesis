# UAV Edge VLM - Project Notes

_Last updated: July 2026_

---

## System Overview

Fine-tuned Qwen2.5-VL-3B on VisDrone aerial captions using QLoRA (r=8, bf16, 1,000 samples, 3 epochs). Deployed on a Jetson Orin NX 16GB via llama.cpp (native build, CUDA SM 8.7) using Q4+imatrix quantization.

---

## HPC Environment (University of Twente EWI Cluster)

- Head node: `hpc-head1.ewi.utwente.nl`, user `s3335739`, venv `~/uav_env`
- Only `main-gpu` partition works for this account; all others reject with "account not permitted"
- Compute nodes have no internet → always set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `local_files_only=True`
- SSL fix required in every job: `export LD_LIBRARY_PATH=/home/s3335739/uav_finetune/libs:$LD_LIBRARY_PATH`
- Model class: `Qwen2_5_VLForConditionalGeneration` (not `Qwen2VL...`)
- Always use bfloat16 → float16 produces NaN loss on this model

---

## Jetson Deployment Notes

- Python/transformers route is a dead end on JetPack 5.1.3 (Python 3.8 locked to CUDA PyTorch wheel; transformers ≥4.49 needs Python ≥3.9) → compile llama.cpp natively, run as persistent HTTP server
- Ollama corrupts fine-tuned Qwen2.5-VL GGUFs → use llama-mtmd-cli directly, not Ollama
- Naive Q4/Q8 quantization erases LoRA deltas → use Q4+imatrix only
- Per-frame model reload costs ~32s → use persistent server mode (~9s)

---

## Troubleshooting

- GPU state corrupts after many hard kills (`pkill -9`) → reboot Jetson if inference output suddenly degrades
- imgs 22 & 23 fail to parse (model runs out of tokens on very dense scenes, GT 162 and 101) → increase `n_predict` or accept as limitation
- Quantization on VisDrone images fails in some configurations, root cause unconfirmed → use Q4+imatrix, which is verified working