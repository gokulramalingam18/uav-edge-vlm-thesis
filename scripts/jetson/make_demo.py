import os, csv, time, base64, json, re, urllib.request, io
from PIL import Image, ImageDraw, ImageFont

URL = "http://127.0.0.1:8080/v1/chat/completions"
PROMPT = ("You are analyzing an aerial UAV image. "
          "Respond in this exact format: "
          "Aerial UAV view showing [count] [object], [count] [object]. "
          "Total [N] objects detected. [scene type]. "
          "Be concise and follow the format exactly.")
TESTDIR = os.path.expanduser("~/uav_thesis/pareto_testset")
OUTDIR = os.path.expanduser("~/uav_thesis/demo_annotated_v2")
os.makedirs(OUTDIR, exist_ok=True)

try:
    font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
except:
    font_large = font_small = ImageFont.load_default()

labels = {}
with open(os.path.join(TESTDIR, "labels.csv")) as f:
    for row in csv.DictReader(f):
        labels[row["image"]] = int(row["gt_total"])

def encode(path):
    im = Image.open(path).convert("RGB").resize((720, 405))
    buf = io.BytesIO(); im.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode(), im

def parse_total(text):
    for pat in [r"Total\s+(\d+)\s+objects", r"Total:\s*(\d+)",
                r"(\d+)\s+objects detected", r"total of\s+(\d+)"]:
        m = re.search(pat, text, re.IGNORECASE)
        if m: return int(m.group(1))
    return None

results = []
for name, gt in sorted(labels.items()):
    b64, im = encode(os.path.join(TESTDIR, name))
    payload = {"messages":[{"role":"user","content":[
        {"type":"text","text":PROMPT},
        {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}]}],
        "n_predict":128,"temperature":0.0,"repeat_penalty":1.1}
    t0 = time.time()
    out = json.loads(urllib.request.urlopen(
        urllib.request.Request(URL, data=json.dumps(payload).encode(),
        headers={"Content-Type":"application/json"}), timeout=60).read())
    el = time.time()-t0
    cap = out["choices"][0]["message"]["content"].strip()
    pred = parse_total(cap)
    err = abs(pred-gt) if pred else None

    W, H = im.size
    panel_h = 120
    canvas = Image.new("RGB", (W, H+panel_h), (15,15,25))
    canvas.paste(im, (0,0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0,H,W,H+panel_h], fill=(15,15,25))
    draw.text((10,H+8), f"Ground Truth: {gt} objects", fill=(80,220,80), font=font_large)
    if pred is not None:
        col = (80,220,80) if err<=10 else (255,180,0) if err<=30 else (255,80,80)
        draw.text((10,H+40), f"Model Pred:   {pred} objects  |  Error: {err}", fill=col, font=font_large)
    else:
        draw.text((10,H+40), "Model Pred:   (format not matched)", fill=(255,80,80), font=font_large)
    draw.text((10,H+80), f"Latency: {el:.1f}s  |  Q4+imatrix, 1024tok  |  {name}", fill=(160,160,180), font=font_small)
    canvas.save(os.path.join(OUTDIR, f"demo_{name}"))
    results.append((name,gt,pred,err,el))
    print(f"  {name}: gt={gt} pred={pred} err={err} {el:.1f}s")

parsed = [r for r in results if r[3] is not None]
print(f"\nDone. Parsed {len(parsed)}/{len(results)}. "
      f"MAE={sum(r[3] for r in parsed)/len(parsed):.1f}" if parsed else "\nDone.")
