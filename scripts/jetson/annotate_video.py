import os, glob, subprocess, time, textwrap
from PIL import Image, ImageDraw, ImageFont

BIN = os.path.expanduser("~/uav_thesis/llama_cpp_jetson/build/bin/llama-mtmd-cli")
MODEL = os.path.expanduser("~/uav_thesis/models/gguf/qwen_uav_text.gguf")
MMPROJ = os.path.expanduser("~/uav_thesis/models/gguf/base_mmproj.gguf")
PROMPT = "Describe what the UAV is observing in this aerial image. Identify objects and estimate the scene type. Be concise."
FRAMES = sorted(glob.glob(os.path.expanduser("~/uav_thesis/demo_frames/*.jpg")))
OUTDIR = os.path.expanduser("~/uav_thesis/demo_annotated")
CSV = os.path.expanduser("~/uav_thesis/results/demo_latency.csv")

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
except:
    font = ImageFont.load_default()

rows = ["frame,seconds"]
for i, img_path in enumerate(FRAMES):
    name = os.path.basename(img_path)
    t0 = time.time()
    out = subprocess.run(
        [BIN, "-m", MODEL, "--mmproj", MMPROJ, "--image", img_path,
         "--image-min-tokens", "1024", "-p", PROMPT, "-n", "128"],
        capture_output=True, text=True)
    elapsed = time.time() - t0
    caption = out.stdout.strip().split("\n")[-1] if out.stdout.strip() else "(no output)"
    rows.append(f"{name},{elapsed:.2f}")
    print(f"  {name}: {elapsed:.1f}s -> {caption[:60]}...")

    im = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    W, H = im.size
    wrapped = textwrap.fill(caption, width=70)
    lines = wrapped.split("\n")
    box_h = len(lines) * 28 + 16
    draw.rectangle([0, H - box_h, W, H], fill=(0, 0, 0))
    y = H - box_h + 8
    for line in lines:
        draw.text((12, y), line, fill=(255, 255, 255), font=font)
        y += 28
    im.save(os.path.join(OUTDIR, f"annotated_{i:04d}.jpg"))

with open(CSV, "w") as f:
    f.write("\n".join(rows))
print(f"\nDONE. {len(FRAMES)} frames annotated. Latency -> {CSV}")
