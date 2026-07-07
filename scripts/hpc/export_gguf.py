import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
LORA_DIR = os.path.expanduser("~/uav_finetune/output/best_model")
MERGED_DIR = os.path.expanduser("~/uav_finetune/output/merged_model")
os.makedirs(MERGED_DIR, exist_ok=True)

print("Loading base model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
    local_files_only=True
)
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, LORA_DIR)
print("Merging weights...")
model = model.merge_and_unload()
print("Saving merged model...")
model.save_pretrained(MERGED_DIR, safe_serialization=True)
processor = AutoProcessor.from_pretrained(BASE_MODEL, local_files_only=True)
processor.save_pretrained(MERGED_DIR)
print(f"Merged model saved to {MERGED_DIR}")
