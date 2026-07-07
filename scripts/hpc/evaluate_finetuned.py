import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json, re
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
LORA_DIR = os.path.expanduser("~/uav_finetune/output/best_model")
VAL_JSON = os.path.expanduser("~/uav_finetune/dataset/visdrone_captions/val.json")
OUT_CSV = os.path.expanduser("~/uav_finetune/eval_results.csv")

print("CUDA:", torch.cuda.is_available(), flush=True)
processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True, local_files_only=True)

with open(VAL_JSON) as f:
    data = json.load(f)
print("Evaluating", len(data), "images", flush=True)

print("Loading fine-tuned model...", flush=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True, local_files_only=True)
model = PeftModel.from_pretrained(model, LORA_DIR)
model.eval()

def total_count(text):
    m = re.search(r"Total\s+(\d+)\s+objects", text)
    return int(m.group(1)) if m else None

def caption(image, question):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": question}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    trimmed = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

abs_errors = []
parsed_ok = 0
rows = ["idx,gt_total,pred_total,abs_error"]
for i, item in enumerate(data):
    try:
        image = Image.open(item["image"]).convert("RGB")
        question = item["conversations"][0]["value"].replace("<image>\n", "")
        gt = total_count(item["conversations"][1]["value"])
        pred = total_count(caption(image, question))
        del image
        torch.cuda.empty_cache()
        if gt is not None and pred is not None:
            err = abs(gt - pred)
            abs_errors.append(err)
            parsed_ok += 1
            rows.append(str(i)+","+str(gt)+","+str(pred)+","+str(err))
        else:
            rows.append(str(i)+","+str(gt)+","+str(pred)+",NA")
    except Exception as e:
        print("ERR idx", i, ":", e, flush=True)
        rows.append(str(i)+",ERR,ERR,NA")
    if (i+1) % 50 == 0:
        print(" ", i+1, "/", len(data), "done", flush=True)

with open(OUT_CSV, "w") as f:
    f.write("\n".join(rows))

n = len(abs_errors)
mae = sum(abs_errors)/n if n else float("nan")
within5 = sum(1 for e in abs_errors if e <= 5)/n*100 if n else 0
within10 = sum(1 for e in abs_errors if e <= 10)/n*100 if n else 0
print("="*50, flush=True)
print("Images evaluated:", len(data), flush=True)
print("Both counts parsed:", parsed_ok, flush=True)
print("MAE (total object count): %.2f" % mae, flush=True)
print("Within +/-5: %.1f%%   Within +/-10: %.1f%%" % (within5, within10), flush=True)
print("CSV saved:", OUT_CSV, flush=True)
print("="*50, flush=True)
