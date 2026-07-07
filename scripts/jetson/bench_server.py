import os, glob, time, base64, json, urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"
PROMPT = "Describe what the UAV is observing in this aerial image. Identify objects and estimate the scene type. Be concise."
FRAMES = sorted(glob.glob(os.path.expanduser("~/uav_thesis/demo_frames/*.jpg")))

def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

times = []
for img in FRAMES:
    name = os.path.basename(img)
    payload = {
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64(img)}"}}
        ]}],
        "n_predict": 64
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    resp = urllib.request.urlopen(req)
    out = json.loads(resp.read())
    el = time.time() - t0
    times.append(el)
    cap = out["choices"][0]["message"]["content"].strip().split("\n")[-1]
    print(f"  {name}: {el:.2f}s -> {cap[:55]}")

print(f"\nMEAN {sum(times)/len(times):.2f}s  MIN {min(times):.2f}s  MAX {max(times):.2f}s")
