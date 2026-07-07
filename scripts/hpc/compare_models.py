import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import json
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
LORA_DIR = os.path.expanduser("~/uav_finetune/output/best_model")
VAL_JSON = os.path.expanduser("~/uav_finetune/dataset/visdrone_captions/val.json")
N = 5

print(f"CUDA: {torch.cuda.is_available()}")
processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True, local_files_only=True)

with open(VAL_JSON) as f:
    data = json.load(f)[:N]

def caption(model, image, question):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": question}
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    trimmed = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

print("\nLoading BASE model...")
base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True, local_files_only=True)
base.eval()

print("Loading FINE-TUNED model...")
ft = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True, local_files_only=True)
ft = PeftModel.from_pretrained(ft, LORA_DIR)
ft.eval()

for i, item in enumerate(data):
    image = Image.open(item["image"]).convert("RGB")
    question = item["conversations"][0]["value"].replace("<image>\n", "")
    truth = item["conversations"][1]["value"]
    print("\n" + "="*70)
    print(f"IMAGE {i+1}: {os.path.basename(item['image'])}")
    print(f"QUESTION: {question}")
    print(f"\n[GROUND TRUTH]\n{truth}")
    print(f"\n[BASE MODEL]\n{caption(base, image, question)}")
    print(f"\n[FINE-TUNED]\n{caption(ft, image, question)}")

print("\n" + "="*70)
print("DONE")
