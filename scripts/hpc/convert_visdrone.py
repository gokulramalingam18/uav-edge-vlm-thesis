import os
import json
import random
from collections import Counter

# Paths
ANNO_DIR = os.path.expanduser("~/uav_finetune/dataset/visdrone/VisDrone2019-DET-train/annotations")
IMG_DIR = os.path.expanduser("~/uav_finetune/dataset/visdrone/VisDrone2019-DET-train/images")
OUTPUT_DIR = os.path.expanduser("~/uav_finetune/dataset/visdrone_captions")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Category mapping
CATEGORIES = {
    1: "pedestrian", 2: "person", 3: "bicycle", 4: "car",
    5: "van", 6: "truck", 7: "tricycle", 8: "awning-tricycle",
    9: "bus", 10: "motorcycle", 11: "other"
}

def annotations_to_caption(anno_file):
    objects = Counter()
    with open(anno_file) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            score = int(parts[4])
            category = int(parts[5])
            if score == 0:  # ignored region
                continue
            if category in CATEGORIES:
                objects[CATEGORIES[category]] += 1

    if not objects:
        return None

    # Build natural language caption
    parts = []
    for obj, count in sorted(objects.items(), key=lambda x: -x[1]):
        if count == 1:
            parts.append(f"1 {obj}")
        else:
            parts.append(f"{count} {obj}s")

    total = sum(objects.values())
    caption = f"Aerial UAV view showing {', '.join(parts)}. Total {total} objects detected."

    # Add scene context based on dominant objects
    if objects.get("car", 0) + objects.get("van", 0) + objects.get("truck", 0) > 5:
        caption += " Dense vehicle traffic visible."
    if objects.get("pedestrian", 0) + objects.get("person", 0) > 3:
        caption += " Multiple pedestrians on ground."
    if objects.get("car", 0) > 10:
        caption += " High density urban or parking area."

    return caption

# Process all annotations
dataset = []
anno_files = sorted(os.listdir(ANNO_DIR))
print(f"Processing {len(anno_files)} annotation files...")

for fname in anno_files:
    if not fname.endswith(".txt"):
        continue

    img_name = fname.replace(".txt", ".jpg")
    img_path = os.path.join(IMG_DIR, img_name)
    anno_path = os.path.join(ANNO_DIR, fname)

    if not os.path.exists(img_path):
        continue

    caption = annotations_to_caption(anno_path)
    if not caption:
        continue

    dataset.append({
        "image": img_path,
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nDescribe what the UAV is observing in this aerial image. Identify objects and estimate the scene type. Be concise."
            },
            {
                "from": "gpt",
                "value": caption
            }
        ]
    })

# Shuffle and split 90/10
random.shuffle(dataset)
split = int(len(dataset) * 0.9)
train_data = dataset[:split]
val_data = dataset[split:]

# Save
with open(os.path.join(OUTPUT_DIR, "train.json"), "w") as f:
    json.dump(train_data, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "val.json"), "w") as f:
    json.dump(val_data, f, indent=2)

print(f"\nDataset conversion complete:")
print(f"  Total samples : {len(dataset)}")
print(f"  Train samples : {len(train_data)}")
print(f"  Val samples   : {len(val_data)}")
print(f"  Saved to      : {OUTPUT_DIR}")

# Show sample caption
print(f"\nSample caption:")
print(dataset[0]["conversations"][1]["value"])
