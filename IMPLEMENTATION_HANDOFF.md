# UAV Edge VLM Thesis — Implementation Handoff

_Last updated: 2026-06-19. Save this file; open it next session to resume._

---

## WHERE YOU ARE

Fine-tuning is **complete and validated**. You have a working LoRA adapter that
demonstrably improves UAV scene understanding. Everything below is what remains.

**Key artifact:** `~/uav_finetune/output/best_model/adapter_model.safetensors` (7.1 MB,
epoch-2 checkpoint, val_loss 0.3948).

**Proven results:**
- Loss curve: train 0.5026 → 0.3734 → 0.3446; val 0.4145 → 0.3948 → 0.4010
- Fine-tuned model produces structured UAV inventory output (vs base model's free prose)
- Hallucination reduced (base invented night scenes / accidents; fine-tuned stays grounded)
- Counting: approximate, not exact (Img3 exact 55; Img4 60v67; Img5 82v73; Img1 over-counts)

**Written up:** `Thesis_Chapter5_FineTuning_Results.docx` (delivered this session).
**Also have:** `Thesis_Chapter1_Introduction.docx`, `UAV_Thesis_Progress.pptx`.

---

## ENVIRONMENT REMINDERS (HPC)

- Head node: `hpc-head1.ewi.utwente.nl`, user `s3335739`, venv `~/uav_env`
- **Only `main-gpu` partition works** — your account is group `ps`; students/bss/smm/tfe/
  dacs-scs-gpu all reject with "account not permitted". Don't retry them.
- Compute nodes have **no internet** → always set `HF_HUB_OFFLINE=1`,
  `TRANSFORMERS_OFFLINE=1`, and `local_files_only=True`. Base model is cached on head node.
- SSL fix needed in every job: `export LD_LIBRARY_PATH=/home/s3335739/uav_finetune/libs:$LD_LIBRARY_PATH`
- Model class is **`Qwen2_5_VLForConditionalGeneration`** (NOT Qwen2VL...).
- Use **bfloat16**, never float16 (float16 → NaN loss on this model).
- Edit scripts via head-node python heredocs + `ast.parse` validation, NOT nano
  (nano repeatedly caused tab/space TabErrors).

---

## IMPLEMENTATION ROADMAP (in priority order)

### STEP 1 — Quantitative evaluation at scale  ← DO THIS FIRST
**Why:** Converts today's 5-image qualitative comparison into a hard benchmark =
your strongest RQ4 evidence.

**What:** Extend `compare_models.py` to run the fine-tuned model over the full 648-image
val set and compute objective metrics:
- Mean Absolute Error (MAE) on **total object count** (fine-tuned vs ground truth)
- Per-category count error (cars, pedestrians, etc.)
- Optionally a caption-similarity metric (e.g. BLEU/ROUGE vs ground-truth caption)

**How:** New script `evaluate_finetuned.py` that loops all 648 val items, parses the
integer counts out of both the model output and the ground-truth string (regex on
"Total N objects"), accumulates errors, prints summary stats. Run as a ~30-min GPU job
on main-gpu (reuse the compare.slurm pattern, bump --time to 01:00:00).

**Output for thesis:** a single MAE number + a per-category error table for Chapter 5.4.4,
replacing the "approximate" qualitative claim with a measured one.

---

### STEP 2 — Merge the adapter for deployment
**Why:** Deployment needs base + adapter combined into one model.

**What:** Run the (already-fixed) merge script:
```bash
sbatch a slurm wrapper around: python3 ~/uav_finetune/scripts/export_gguf.py
```
(`export_gguf.py` was corrected this session: right model class, offline mode, bf16.
It writes `~/uav_finetune/output/merged_model/`.) Needs ~24G RAM; can run CPU-only but
a short GPU job is faster.

**Verify:** after merge, `ls -lh ~/uav_finetune/output/merged_model/` should show
`model-*.safetensors` shards (~6 GB total) + config + processor files.

---

### STEP 3 — Deploy to Jetson via Python/transformers (NOT GGUF)
**Why this path:** llama.cpp / GGUF support for Qwen2.5-VL **vision** is currently broken
(confirmed: fine-tuned-then-merged Qwen2.5-VL GGUFs produce garbage "@@@" output; the
vision encoder needs a separate mmproj surgery step your old convert script doesn't do).
Using transformers preserves the vision pathway and gives a like-for-like comparison
with your baseline (which also ran in Python on the Jetson).

**What:**
1. `scp` the `merged_model/` directory from HPC → Jetson `~/uav_thesis/models/finetuned/`
   (it's ~6 GB; transfer over the network or via the Windows box like the dataset).
2. On the Jetson, check it loads: Python 3.8 + transformers. **Watch out** — the Jetson
   has older torch/transformers (Python 3.8, CUDA 11.4); you may need to confirm the
   installed transformers version supports Qwen2.5-VL. If not, this becomes a sub-task
   (upgrade transformers in a Jetson venv, or run inference in fp16/int8 to fit memory).
3. Load with `Qwen2_5_VLForConditionalGeneration.from_pretrained(..., torch_dtype=float16,
   device_map="auto")` and run one test caption to confirm vision works on-device.

**Risk flag:** Jetson memory (16 GB shared) + a 3B model in fp16 (~6 GB) should fit, but
quantization (int8/4-bit via bitsandbytes) may be needed. bitsandbytes on Jetson/CUDA 11.4
was problematic earlier — if int8 is required and won't build, fall back to fp16 and
smaller batch / lower image resolution.

---

### STEP 4 — On-device latency comparison (the thesis payoff)
**Why:** Directly tests your central finding — "latency is dominated by reasoning depth,
not visual complexity" — in the fine-tuned setting. The structured, shorter output of the
fine-tuned model should generate fewer tokens → predicts LOWER latency than the base model.

**What:** Re-run your Section 5.1 latency experiment harness with the fine-tuned model on
the same UAV footage frames used for the baseline. Log per-frame latency + output. Compare:
- baseline avg 23.0 s vs fine-tuned avg (hypothesis: lower, because less free-form reasoning)
- output token count base vs fine-tuned (your reasoning-depth metric)

**Output for thesis:** the latency comparison table + the confirmation (or refutation) that
fine-tuning's constrained output reduces inference time on edge — closing the loop between
RQ3 (latency drivers) and RQ4 (fine-tuning).

---

### STEP 5 — Strengthen the model (optional, if time)
- Retrain on the **full 5,823-sample** dataset (vs the 1,000 subset) → better counting,
  less overfitting (the epoch-3 val uptick).
- Mitigate **over-generation** (Img1 47 vs 27): fewer epochs, or prompt tweak to discourage
  speculative enumeration, or balance category frequencies in the training captions.

---

## RECOMMENDED ORDER & RATIONALE
1. **Step 1 first** — pure HPC, no new risk, produces citable numbers immediately.
2. **Step 2** — quick, unblocks deployment.
3. **Step 3** — highest uncertainty (Jetson software stack); budget time for the
   transformers/quantization sub-tasks.
4. **Step 4** — the headline experiment; needs Step 3 working first.
5. **Step 5** — only if Steps 1–4 leave time.

## OPEN RISKS TO WATCH
- Jetson transformers version may not support Qwen2.5-VL → may need a venv upgrade.
- Jetson memory may force quantization; bitsandbytes on CUDA 11.4 was fragile.
- If on-device fp16 is too slow to be usable, that itself is a thesis result (edge
  hardware limits), not a failure — report it.
