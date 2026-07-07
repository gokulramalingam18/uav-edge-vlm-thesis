import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim import AdamW
from tqdm import tqdm

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
TRAIN_JSON = os.path.expanduser("~/uav_finetune/dataset/visdrone_captions/train.json")
VAL_JSON = os.path.expanduser("~/uav_finetune/dataset/visdrone_captions/val.json")
OUTPUT_DIR = os.path.expanduser("~/uav_finetune/output")
EPOCHS = 3
LR = 2e-4
MAX_SAMPLES = 1000
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"CUDA: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"Model: {MODEL_NAME}")

class VisDroneDataset(Dataset):
    def __init__(self, json_path, max_samples=None):
        with open(json_path) as f:
            self.data = json.load(f)
        if max_samples:
            self.data = self.data[:max_samples]
        print(f"Loaded {len(self.data)} samples")
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image"]).convert("RGB")
        question = item["conversations"][0]["value"].replace("<image>\n", "")
        answer = item["conversations"][1]["value"]
        return image, question, answer

print("\nLoading model...")
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True, local_files_only=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"], bias="none"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

train_dataset = VisDroneDataset(TRAIN_JSON, MAX_SAMPLES)
val_dataset = VisDroneDataset(VAL_JSON, 100)
optimizer = AdamW(model.parameters(), lr=LR)
print(f"\nTraining: {len(train_dataset)} samples, {EPOCHS} epochs\n")

def build_inputs_and_labels(image, question, answer):
    # Full conversation (prompt + answer) for model input
    full_msgs = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}
        ]},
        {"role": "assistant", "content": answer}
    ]
    full_text = processor.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
    inputs = processor(text=[full_text], images=[image], return_tensors="pt", padding=True).to("cuda")

    # Prompt-only (no answer) to find how many tokens precede the answer
    prompt_msgs = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}
        ]}
    ]
    prompt_text = processor.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    prompt_inputs = processor(text=[prompt_text], images=[image], return_tensors="pt", padding=True)
    prompt_len = prompt_inputs["input_ids"].shape[1]

    labels = inputs["input_ids"].clone()
    # Mask everything up to and including the prompt; train only on answer tokens
    labels[:, :prompt_len] = -100
    # Also mask any pad tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100
    return inputs, labels

best_val_loss = float("inf")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    count = 0
    for image, question, answer in tqdm(train_dataset, desc=f"Epoch {epoch+1}"):
        try:
            inputs, labels = build_inputs_and_labels(image, question, answer)
            n_valid = (labels != -100).sum().item()
            if n_valid == 0:
                continue
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            if count == 0:
                print(f"DIAG n_valid_labels={n_valid} total={labels.numel()} loss={loss.item():.4f}", flush=True)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"SKIP nan/inf loss", flush=True)
                optimizer.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            count += 1
        except Exception as e:
            print(f"SAMPLE ERROR: {e}", flush=True)
            continue
    avg_train_loss = total_loss / max(count, 1)

    model.eval()
    val_loss = 0.0
    val_count = 0
    with torch.no_grad():
        for image, question, answer in tqdm(val_dataset, desc="Validating"):
            try:
                inputs, labels = build_inputs_and_labels(image, question, answer)
                n_valid = (labels != -100).sum().item()
                if n_valid == 0:
                    continue
                outputs = model(**inputs, labels=labels)
                if torch.isnan(outputs.loss) or torch.isinf(outputs.loss):
                    continue
                val_loss += outputs.loss.item()
                val_count += 1
            except Exception as e:
                print(f"VAL ERROR: {e}", flush=True)
                continue
    avg_val_loss = val_loss / max(val_count, 1)

    print(f"\nEpoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f} (trained on {count} samples)")
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model.save_pretrained(os.path.join(OUTPUT_DIR, "best_model"))
        processor.save_pretrained(os.path.join(OUTPUT_DIR, "best_model"))
        print(f"Best model saved (val_loss={best_val_loss:.4f})")

print("\nTraining complete.")
