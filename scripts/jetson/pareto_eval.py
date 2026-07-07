import os, csv, time, base64, json, re, urllib.request, sys, io
from PIL import Image

URL = "http://127.0.0.1:8080/v1/chat/completions"
PROMPT = ("Describe what the UAV is observing in this aerial image. "
          "List each object type with its count, then state the total "
          "number of objects detected, then the scene type. Be concise.")
TESTDIR = os.path.expanduser("~/uav_thesis/pareto_testset")
POINT_NAME = sys.argv[1] if len(sys.argv) > 1 else "point"

labels = {}
with open(os.path.join(TESTDIR, "labels.csv")) as f:
    r = csv.DictReader(f)
    for row in r:
        labels[row["image"]] = int(row["gt_total"])

def encode(path, size=(720,405)):
    im = Image.open(path).convert("RGB")
    im = im.resize(size)
    buf = io.BytesIO()
    im.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()

def parse_total(text):
    m = re.search(r"Total\s+(\d+)\s+objects", text)
    return int(m.group(1)) if m else None

times, errs, rows = [], [], []
for name, gt in sorted(labels.items()):
    img = os.path.join(TESTDIR, name)
    payload = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode(img)}"}}]}],
        "n_predict": 128, "temperature": 0.0}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    out = json.loads(urllib.request.urlopen(req).read())
    el = time.time() - t0
    cap = out["choices"][0]["message"]["content"].strip()
    pred = parse_total(cap)
    err = abs(pred - gt) if pred is not None else None
    times.append(el)
    if err is not None: errs.append(err)
    rows.append((name, gt, pred, err, el))
    print(f"  {name}: gt={gt} pred={pred} err={err} time={el:.1f}s")

mean_lat = sum(times)/len(times)
mean_err = sum(errs)/len(errs) if errs else float('nan')
print(f"\n=== {POINT_NAME} ===")
print(f"Mean latency: {mean_lat:.2f}s  |  MAE: {mean_err:.2f}  |  Parsed: {len(errs)}/{len(labels)}")

out_csv = os.path.expanduser(f"~/uav_thesis/results/pareto_{POINT_NAME}.csv")
with open(out_csv, "w") as f:
    f.write("image,gt,pred,err,seconds\n")
    for row in rows:
        f.write(",".join(str(x) for x in row) + "\n")
print(f"Saved -> {out_csv}")
